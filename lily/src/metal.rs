//! Metal device context: pipeline compilation/caching, buffer allocation, and
//! serial or concurrent compute passes that batch kernel dispatches into one
//! command buffer.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::tensor::Tensor;

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct Kernel {
    pub pipeline: Pipeline,
}

impl Kernel {
    /// The SIMD width this pipeline executes at. Apple's own porting guidance is to
    /// read it from the pipeline rather than assume 32, and a simdgroup-scoped
    /// intrinsic has to be issued by exactly one simdgroup's worth of threads for the
    /// call to be convergent.
    pub fn thread_execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    /// Bytes of statically reserved threadgroup memory reported by the pipeline.
    pub fn static_threadgroup_memory_length(&self) -> usize {
        self.pipeline.staticThreadgroupMemoryLength()
    }
}

/// The MSL language version a kernel source requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MslVersion {
    /// Metal 3.1 (macOS 14+): the baseline for every kernel (`bfloat`).
    V3_1,
    /// Metal 4.0 (macOS 26+): tensor ops / MetalPerformancePrimitives (the
    /// neural-accelerator matmul path on Apple GPU family 10+). On Apple GPU
    /// families 7-9, TensorOps use optimized shader implementations instead.
    /// Lily's BF16 tensor kernels require macOS 26.1+. Compilation fails where
    /// Metal 4 is unavailable, so callers that support older environments must
    /// keep a 3.1 fallback.
    V4_0,
    /// Metal 4.1 (macOS 27+). objc2-metal 0.3.2 predates the named constant,
    /// but MTLLanguageVersion is an open integer wrapper and the SDK value is
    /// `(4 << 16) + 1`.
    V4_1,
}

impl MslVersion {
    fn language_version(self) -> MTLLanguageVersion {
        match self {
            MslVersion::V3_1 => MTLLanguageVersion::Version3_1,
            MslVersion::V4_0 => MTLLanguageVersion::Version4_0,
            MslVersion::V4_1 => MTLLanguageVersion((4 << 16) + 1),
        }
    }
}

/// Largest `params` slice any dispatch passes. Bounds the stack array
/// [`ComputePass::dispatch_at`] validates into; exceeding it is an error rather
/// than a silent heap fallback, so the bound stays honest.
const MAX_KERNEL_PARAMS: usize = 16;

/// Minimum Apple GPU family with Metal 4 support on Apple Silicon.
/// Family 7 corresponds to M1-class GPUs.
const MIN_APPLE_GPU_FAMILY: i64 = 7;

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Keyed by function name and MSL version.
    pipelines: Mutex<HashMap<(&'static str, MslVersion), Pipeline>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device found"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("failed to create command queue"))?;
        let ctx = Self { device, queue, pipelines: Mutex::new(HashMap::new()) };
        // Metal 4 TensorOps are portable across Apple Silicon. Family 10+ can
        // execute them on native GPU Neural Accelerators; families 7-9 use
        // optimized shader implementations. Runtime pipeline compilation is
        // still the final capability check for the installed macOS version.
        let family = ctx.apple_gpu_family();
        ensure!(
            family >= MIN_APPLE_GPU_FAMILY,
            "lily needs Apple GPU family {MIN_APPLE_GPU_FAMILY} or later \
             (M1-class or newer); this device reports family {family}"
        );
        Ok(ctx)
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// True when Metal reports Apple GPU family 10 or newer, where TensorOps
    /// can use the per-GPU-core Neural Accelerators introduced with M5.
    /// Families 7-9 remain supported but return false here.
    pub fn has_native_tensor_acceleration(&self) -> bool {
        self.apple_gpu_family() >= 10
    }

    /// Compiles MSL source through the framework's compiler (the same
    /// front-end an offline `xcrun metal` uses), reporting diagnostics on
    /// failure.
    pub fn compile_library(
        &self,
        source: &str,
        version: MslVersion,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(version.language_version());
        self.device
            .newLibraryWithSource_options_error(
                &NSString::from_str(source),
                Some(&options),
            )
            .map_err(|e| anyhow!("failed to compile Metal library: {e:?}"))
    }

    /// Creates the compute pipeline for `fn_name` from `library`.
    pub fn pipeline_from_library(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        fn_name: &str,
    ) -> Result<Pipeline> {
        let function =
            library.newFunctionWithName(&NSString::from_str(fn_name)).ok_or_else(
                || anyhow!("kernel function '{fn_name}' not found in source"),
            )?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("failed to create pipeline for {fn_name}: {e:?}"))
    }

    /// Returns the compute pipeline for `fn_name`, compiling `source` on first
    /// use. Kernels are compiled from source at runtime, so there is no offline
    /// `.metal` -> `.metallib` step in the build.
    pub fn pipeline(
        &self,
        fn_name: &'static str,
        source: &str,
        version: MslVersion,
    ) -> Result<Kernel> {
        let mut cache = self
            .pipelines
            .lock()
            .map_err(|e| anyhow!("pipeline cache lock poisoned: {e}"))?;
        if let Some(p) = cache.get(&(fn_name, version)) {
            return Ok(Kernel { pipeline: p.clone() });
        }

        let library = self
            .compile_library(source, version)
            .with_context(|| format!("for {fn_name}"))?;
        let pipeline = self.pipeline_from_library(&library, fn_name)?;
        cache.insert((fn_name, version), pipeline.clone());
        Ok(Kernel { pipeline })
    }

    /// Allocates a zero-initialized shared-storage buffer.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let buf = self
            .device
            .newBufferWithLength_options(
                len.max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| anyhow!("failed to allocate {len}-byte buffer"))?;
        // Metal does not guarantee new buffer contents; callers rely on zeros.
        unsafe { core::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, len) };
        Ok(buf)
    }

    pub fn new_buffer_with_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let buf = self.new_buffer(bytes.len())?;
        // SAFETY: the buffer was just allocated with exactly `bytes.len()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(buf)
    }

    /// Starts a compute pass. Dispatches encoded on one pass execute serially
    /// The serial compute pass used by prefill and utility kernels.
    pub fn begin(&self) -> Result<ComputePass<'_>> {
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let encoder = cmd
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("failed to create compute command encoder"))?;
        Ok(ComputePass {
            _ctx: std::marker::PhantomData,
            cmd,
            encoder: RefCell::new(encoder),
            ended: Cell::new(false),
        })
    }

    /// A compute pass whose single encoder uses `MTLDispatchType::Concurrent` —
    /// Metal drops the serial encoder's implicit ordering, so dispatches may
    /// overlap and the caller owns every hazard via
    /// [`ComputePass::memory_barrier`] at dependency boundaries. Used by the
    /// concurrent decode path.
    pub fn begin_concurrent(&self) -> Result<ComputePass<'_>> {
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let desc = MTLComputePassDescriptor::computePassDescriptor();
        desc.setDispatchType(MTLDispatchType::Concurrent);
        let encoder = cmd
            .computeCommandEncoderWithDescriptor(&desc)
            .ok_or_else(|| anyhow!("failed to create concurrent compute encoder"))?;
        Ok(ComputePass {
            _ctx: std::marker::PhantomData,
            cmd,
            encoder: RefCell::new(encoder),
            ended: Cell::new(false),
        })
    }

    /// Highest supported Apple GPU family number known to this build:
    /// 10=M5+, 9=M3/M4, 8=M2, 7=M1. Metal exposes no direct query for whether
    /// TensorOps are using native Neural Accelerators, so the family number is
    /// the architecture-policy signal.
    pub fn apple_gpu_family(&self) -> i64 {
        (1..=10i64)
            .rev()
            .find(|&n| {
                self.device.supportsFamily(objc2_metal::MTLGPUFamily(n as isize + 1000))
            })
            .unwrap_or(0)
    }
}

pub struct ComputePass<'a> {
    _ctx: std::marker::PhantomData<&'a MetalContext>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: RefCell<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    ended: Cell<bool>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        // Metal asserts if an encoder is released mid-encoding; close it so an
        // error path (`?` between dispatches) surfaces the error instead of
        // trapping in the destructor. The unfinished command buffer is simply
        // never committed.
        if !self.ended.get() {
            self.encoder.borrow().endEncoding();
        }
    }
}

/// How to map work items onto the GPU for one dispatch.
#[derive(Clone, Copy)]
pub enum Grid {
    /// `dispatchThreads` with a non-uniform grid of exactly this many threads.
    Threads { grid: (usize, usize, usize), threadgroup: (usize, usize, usize) },
    /// `dispatchThreadgroups`: kernels that reduce within a threadgroup need an
    /// exact threadgroup count (e.g. one per output row).
    Threadgroups { groups: (usize, usize, usize), threadgroup: (usize, usize, usize) },
}

impl<'a> ComputePass<'a> {
    /// Encodes one kernel dispatch with a byte offset per buffer binding — used
    /// to bind slices of a larger buffer (e.g. the q/k/v thirds of the fused
    /// GDN projection). Offsets must be 4-byte aligned per Metal's rules.
    pub fn dispatch_at(
        &self,
        kernel: &Kernel,
        buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
        params: &[&[u8]],
        grid: Grid,
    ) -> Result<()> {
        // Validate params before any encoder exists so an error can't leave an
        // encoder open -- but into a stack array, not a Vec. This runs once per
        // dispatch and a decode step submits hundreds of them, so the collect
        // would be a malloc/free pair on the latency-sensitive host path.
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        let mut param_buf =
            [(NonNull::<c_void>::dangling(), 0usize); MAX_KERNEL_PARAMS];
        for (slot, bytes) in param_buf.iter_mut().zip(params) {
            let ptr = NonNull::new(bytes.as_ptr().cast::<c_void>().cast_mut())
                .context("empty kernel param")?;
            *slot = (ptr, bytes.len());
        }
        let param_ptrs = param_buf.get(..params.len()).context("param slice")?;

        encode_dispatch(&self.encoder.borrow(), kernel, buffers, param_ptrs, grid);
        Ok(())
    }

    /// Orders one dependency level before the next on a concurrent encoder.
    pub fn level_barrier(&self, _written: &[&Tensor]) -> Result<()> {
        self.memory_barrier()
    }

    /// Orders all prior dispatches' buffer writes before every subsequent
    /// dispatch — the dependency-level boundary for [`MetalContext::
    /// begin_concurrent`] passes. Serial encoders safely ignore it.
    pub fn memory_barrier(&self) -> Result<()> {
        self.encoder.borrow().memoryBarrierWithScope(MTLBarrierScope::Buffers);
        Ok(())
    }

    /// Ends encoding and submits without blocking; pair with
    /// [`PendingPass::wait`]. Two passes in flight let the host encode step
    /// N+1 while the GPU runs step N (the queue executes them in submit
    /// order).
    pub fn commit(self) -> Result<PendingPass<'a>> {
        self.encoder.borrow().endEncoding();
        self.ended.set(true);
        self.cmd.commit();
        Ok(PendingPass { _ctx: std::marker::PhantomData, cmd: self.cmd.clone() })
    }

    /// Ends encoding, submits the command buffer, and blocks until the GPU
    /// finishes.
    pub fn commit_wait(self) -> Result<()> {
        self.encoder.borrow().endEncoding();
        self.ended.set(true);
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        Ok(())
    }
}

/// A committed-but-unawaited pass. Holding one while encoding the next pass
/// is the decode pipelining primitive; hosts must not read buffers the
/// pending pass writes until [`Self::wait`] returns.
pub struct PendingPass<'a> {
    _ctx: std::marker::PhantomData<&'a MetalContext>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[derive(Clone, Copy, Debug)]
pub struct PassTiming {
    pub gpu_start_secs: f64,
    pub gpu_end_secs: f64,
}

pub struct CompletedPass {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CompletedPass {
    pub fn timing(&self) -> Result<PassTiming> {
        let timing = PassTiming {
            gpu_start_secs: self.cmd.GPUStartTime(),
            gpu_end_secs: self.cmd.GPUEndTime(),
        };
        ensure!(
            timing.gpu_start_secs.is_finite()
                && timing.gpu_end_secs.is_finite()
                && timing.gpu_start_secs > 0.0
                && timing.gpu_end_secs > timing.gpu_start_secs,
            "invalid Metal GPU timestamps: {timing:?}"
        );
        Ok(timing)
    }
}

impl PendingPass<'_> {
    /// Blocks until the GPU finishes this pass.
    pub fn wait(self) -> Result<()> {
        self.cmd.waitUntilCompleted();
        Ok(())
    }

    /// Blocks until completion while retaining the completed command buffer.
    /// Benchmarks query its GPU clock only after their cadence timer ends.
    pub fn wait_retain(self) -> Result<CompletedPass> {
        self.cmd.waitUntilCompleted();
        Ok(CompletedPass { cmd: self.cmd })
    }
}

fn encode_dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    kernel: &Kernel,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    params: &[(NonNull<c_void>, usize)],
    grid: Grid,
) {
    encoder.setComputePipelineState(&kernel.pipeline);
    for (i, (buf, offset)) in buffers.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(buf), *offset, i) };
    }
    for (i, (ptr, len)) in params.iter().enumerate() {
        unsafe { encoder.setBytes_length_atIndex(*ptr, *len, buffers.len() + i) };
    }
    let size =
        |(w, h, d): (usize, usize, usize)| MTLSize { width: w, height: h, depth: d };
    match grid {
        Grid::Threads { grid, threadgroup } => {
            encoder
                .dispatchThreads_threadsPerThreadgroup(size(grid), size(threadgroup));
        }
        Grid::Threadgroups { groups, threadgroup } => {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size(groups),
                size(threadgroup),
            );
        }
    }
}