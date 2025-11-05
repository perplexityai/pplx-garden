use std::collections::{HashMap, HashSet};
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use cuda_lib::Device;
use parking_lot::Mutex;

use crate::{RdmaEngine, api::MemoryRegionHandle, error::Result};

pub struct HostBuffer {
    index: usize,
    ptr: *mut u8,
    length: usize,
    mr_handle: MemoryRegionHandle,
    cache: Arc<Mutex<HostBufferCache>>,
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    pub fn mr_handle(&self) -> MemoryRegionHandle {
        self.mr_handle
    }

    pub fn as_nonnull(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.ptr as *mut c_void) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.length) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        self.cache.lock().free(self.index, self.length);
    }
}

struct HostBufferEntry {
    #[allow(clippy::box_collection)]
    storage: Box<Vec<u8>>,
    mr_handle: MemoryRegionHandle,
}

impl HostBufferEntry {
    fn new(engine: Arc<impl RdmaEngine>, size: usize) -> Result<Self> {
        let mut storage = Box::new(Vec::with_capacity(size));
        unsafe { storage.set_len(size) };

        // Register the memory region
        let buf_base =
            unsafe { NonNull::new_unchecked(storage.as_mut_ptr() as *mut c_void) };
        let mr_handle = engine.register_memory_local(buf_base, size, Device::Host)?;

        Ok(Self { storage, mr_handle })
    }
}

struct HostBufferCache {
    buffers: Vec<HostBufferEntry>,
    free_bufs: HashMap<usize, HashSet<usize>>,
}

impl HostBufferCache {
    fn free(&mut self, index: usize, length: usize) {
        self.free_bufs.entry(length).or_default().insert(index);
    }
}

#[derive(Clone)]
pub struct HostBufferAllocator<E: RdmaEngine> {
    cache: Arc<Mutex<HostBufferCache>>,
    engine: Arc<E>,
}

impl<E: RdmaEngine> HostBufferAllocator<E> {
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HostBufferCache {
                buffers: Vec::new(),
                free_bufs: HashMap::new(),
            })),
            engine,
        }
    }

    pub fn allocate(&self, size: usize) -> Result<HostBuffer> {
        // Try to find a free buffer
        // NOTE(lequn): I don't expect there to be many free buffers, so
        // doing a linear scan for now. If this becomes a problem, we can
        // introduce a proper memory allocator.
        let mut cache = self.cache.lock();
        {
            for (&allocated_size, entries) in cache.free_bufs.iter_mut() {
                if allocated_size < size {
                    continue;
                }
                match entries.iter().next().cloned() {
                    None => continue,
                    Some(index) => {
                        entries.remove(&index);
                        let entry = &cache.buffers[index];
                        return Ok(HostBuffer {
                            index,
                            ptr: entry.storage.as_ptr() as *mut u8,
                            length: allocated_size,
                            mr_handle: entry.mr_handle,
                            cache: self.cache.clone(),
                        });
                    }
                }
            }
        }

        // Otherwise allocate a new buffer. Size up to the next power of two.
        let size = size.next_power_of_two();
        let index = cache.buffers.len();
        cache.buffers.push(HostBufferEntry::new(self.engine.clone(), size)?);
        let entry = &cache.buffers[index];
        Ok(HostBuffer {
            index,
            ptr: entry.storage.as_ptr() as *mut u8,
            length: size,
            mr_handle: entry.mr_handle,
            cache: self.cache.clone(),
        })
    }
}
