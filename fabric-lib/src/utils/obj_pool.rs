use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// A growable, chunked object pool with stable addresses suitable for FFI.
///
/// - Grows by appending fixed-size chunks (each chunk is a separate Box allocation).
/// - Previously returned pointers remain valid for the lifetime of the pool.
/// - Not thread-safe. Wrap in a Mutex if needed.
/// - Safety: You must call `free_and_drop` to run `T`'s destructor if `T: Drop`.
pub struct ObjectPool<T> {
    chunks: Vec<Box<[MaybeUninit<T>]>>,
    chunk_size: usize,

    // Index of next unallocated slot within the current (last) chunk.
    // Ranges from 0..=chunk_size. If == chunk_size we need a new chunk.
    next_in_last: usize,

    // LIFO free list of previously freed slots (as raw pointers).
    free_list: Vec<NonNull<MaybeUninit<T>>>,
}

impl<T> ObjectPool<T> {
    /// Create a pool with a given chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            chunks: Vec::new(),
            chunk_size,
            next_in_last: 0,
            free_list: Vec::with_capacity(chunk_size),
        }
    }

    /// Allocate an **uninitialized** slot. You must initialize it before reading.
    ///
    /// Safety:
    /// - Returned pointer must be written with a valid `T` before any read or `dealloc_init`.
    /// - Pointer remains valid until returned to this pool or the pool is dropped.
    pub unsafe fn alloc_uninit(&mut self) -> NonNull<MaybeUninit<T>> {
        if let Some(p) = self.free_list.pop() {
            return p;
        }
        if self.chunks.is_empty() || self.next_in_last == self.chunk_size {
            // New chunk of MaybeUninit<T> (elements stay uninitialized).
            let new = Box::<[MaybeUninit<T>]>::new_uninit_slice(self.chunk_size);
            let new = unsafe { new.assume_init() };
            self.chunks.push(new);
            self.next_in_last = 0;
        }
        let c = self.chunks.len() - 1;
        let s = self.next_in_last;
        self.next_in_last += 1;
        unsafe { NonNull::new_unchecked(self.chunks[c].as_mut_ptr().add(s)) }
    }

    /// Deallocate a previously allocated pointer and run its destructor.
    ///
    /// Safety:
    /// - `p` must have been returned by `alloc_uninit` of *this* pool,
    ///   and not already deallocated.
    pub unsafe fn free_and_drop(&mut self, p: NonNull<T>) {
        // SAFETY: caller guarantees p originated from this pool and is not freed twice.
        unsafe { std::ptr::drop_in_place(p.as_ptr()) };
        self.free_list
            .push(unsafe { NonNull::new_unchecked(p.as_ptr() as *mut MaybeUninit<T>) });
    }

    /// Deallocate a previously allocated pointer without running its destructor.
    ///
    /// Safety:
    /// - `p` must have been returned by `alloc_uninit` of *this* pool,
    ///   and not already deallocated.
    /// - Only use this if `p` was never initialized or T is POD.
    #[allow(dead_code)]
    pub unsafe fn free_no_drop(&mut self, p: *mut MaybeUninit<T>) {
        debug_assert!(!p.is_null());
        // SAFETY: caller guarantees p originated from this pool and is not freed twice.
        self.free_list.push(unsafe { NonNull::new_unchecked(p) });
    }
}
