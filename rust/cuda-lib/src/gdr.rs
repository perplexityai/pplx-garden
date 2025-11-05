use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr::null_mut,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use crate::CudaError;

type GdrResult<T> = Result<T, CudaError>;

/// Reference-counted GDR context handle.
struct GdrContextHandle {
    handle: gdrapi_sys::gdr_t,
}
unsafe impl Send for GdrContextHandle {}
unsafe impl Sync for GdrContextHandle {}

impl Drop for GdrContextHandle {
    fn drop(&mut self) {
        unsafe { gdrapi_sys::gdr_close(self.handle) };
    }
}

/// Public wrapper around the GDRCopy context.
pub struct GdrCopyContext {
    context: Arc<GdrContextHandle>,
}

fn align_to(ptr: u64, alignment: usize) -> u64 {
    (ptr + alignment as u64 - 1).div_ceil(alignment as u64) * alignment as u64
}

impl GdrCopyContext {
    pub fn new() -> GdrResult<Self> {
        let handle = unsafe { gdrapi_sys::gdr_open() };
        if handle.is_null() {
            return Err(CudaError::GdrCopyError("Failed to create GDR copy handle"));
        }
        Ok(GdrCopyContext { context: Arc::new(GdrContextHandle { handle }) })
    }

    fn alloc_buffer(&self, nbytes: usize) -> GdrResult<GdrBuffer> {
        let mut device_ptr: u64 = 0;
        let page_size: usize = 1 << 16; // 64KB page size
        let bytesize = nbytes.div_ceil(page_size) * page_size;

        if unsafe { cuda_sys::cuMemAlloc(&mut device_ptr, bytesize + page_size) }
            != cuda_sys::CUDA_SUCCESS
        {
            return Err(CudaError::GdrCopyError("Failed to allocate GDR buffer"));
        }

        let aligned_device_ptr = align_to(device_ptr, page_size);

        let context = self.context.clone();

        let g = context.handle;
        let mut mh = gdrapi_sys::gdr_mh_t { h: 0 };

        let ret = unsafe {
            gdrapi_sys::gdr_pin_buffer(g, aligned_device_ptr, bytesize, 0, 0, &mut mh)
        };
        if ret != 0 {
            unsafe { cuda_sys::cuMemFree(device_ptr) };
            return Err(CudaError::GdrCopyError("Failed to pin GDR buffer"));
        }

        let mut mapped_ptr: *mut c_void = null_mut();
        let ret = unsafe { gdrapi_sys::gdr_map(g, mh, &mut mapped_ptr, bytesize) };
        if ret != 0 {
            unsafe {
                gdrapi_sys::gdr_unpin_buffer(g, mh);
                cuda_sys::cuMemFree(device_ptr);
            };
            return Err(CudaError::GdrCopyError("Failed to map GDR buffer"));
        }

        Ok(GdrBuffer {
            device_ptr,
            aligned_device_ptr,
            mapped_ptr,
            bytesize,
            mh,
            context,
        })
    }
}

/// Raw buffer allocated on the CPU, copied using GDRCopy.
struct GdrBuffer {
    device_ptr: u64,
    aligned_device_ptr: u64,
    mapped_ptr: *mut c_void,
    mh: gdrapi_sys::gdr_mh_t,
    bytesize: usize,
    context: Arc<GdrContextHandle>,
}

unsafe impl Send for GdrBuffer {}
unsafe impl Sync for GdrBuffer {}

impl Drop for GdrBuffer {
    fn drop(&mut self) {
        let g = self.context.handle;
        unsafe {
            gdrapi_sys::gdr_unmap(g, self.mh, self.mapped_ptr, self.bytesize);
            gdrapi_sys::gdr_unpin_buffer(g, self.mh);
            cuda_sys::cuMemFree(self.device_ptr);
        };
    }
}

trait GdrRead {
    fn read(mapped_ptr: *mut c_void) -> Self;
}

impl GdrRead for u8 {
    #[inline(always)]
    fn read(mapped_ptr: *mut c_void) -> Self {
        let flag = unsafe { AtomicU8::from_ptr(mapped_ptr as *mut u8) };
        flag.load(Ordering::Acquire)
    }
}

trait GdrWrite {
    fn write(mapped_ptr: *mut c_void, value: Self);
}

impl GdrWrite for u8 {
    #[inline(always)]
    fn write(mapped_ptr: *mut c_void, value: Self) {
        let flag = unsafe { AtomicU8::from_ptr(mapped_ptr as *mut u8) };
        flag.store(value, Ordering::Release);
    }
}

impl GdrBuffer {
    fn get_device_ptr(&self) -> *mut c_void {
        self.aligned_device_ptr as *mut c_void
    }

    #[inline(always)]
    fn read<T: GdrRead>(&self) -> T {
        T::read(self.mapped_ptr)
    }

    #[inline(always)]
    fn write<T: GdrWrite>(&self, value: T) {
        T::write(self.mapped_ptr, value);
    }

    fn copy_to(&self, src: *const c_void, nbytes: usize) {
        unsafe {
            gdrapi_sys::gdr_copy_to_mapping(self.mh, self.mapped_ptr, src, nbytes);
        }
    }
}

/// Byte-flag implemented using GDRCopy.
pub struct GdrFlag {
    buffer: GdrBuffer,
}

impl GdrFlag {
    pub fn new(context: &GdrCopyContext) -> GdrResult<Self> {
        let buffer = context.alloc_buffer(size_of::<u8>())?;
        Ok(GdrFlag { buffer })
    }

    pub fn wait(&self) {
        while !self.is_set() {
            std::hint::spin_loop();
        }
        self.set(false);
    }

    pub fn get_device_ptr(&self) -> *mut u8 {
        self.buffer.get_device_ptr() as *mut u8
    }

    pub fn set(&self, value: bool) {
        self.buffer.write(value as u8);
    }

    fn is_set(&self) -> bool {
        self.buffer.read::<u8>() != 0
    }
}

pub struct GdrVec<T: Sized> {
    buffer: GdrBuffer,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Sized> GdrVec<T> {
    pub fn new(context: &GdrCopyContext, len: usize) -> GdrResult<Self> {
        let buffer = context.alloc_buffer(len * size_of::<T>())?;
        Ok(GdrVec { buffer, len, _marker: std::marker::PhantomData })
    }

    pub fn get_device_ptr(&self) -> *mut T {
        self.buffer.get_device_ptr().cast::<T>()
    }

    pub fn copy(&self, value: &[T]) {
        debug_assert!(value.len() <= self.len);
        self.buffer.copy_to(value.as_ptr() as *const c_void, size_of_val(value));
    }
}
