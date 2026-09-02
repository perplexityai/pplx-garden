//! BF16 matrix multiplication over row-major `[out, in]` weights.

use std::sync::OnceLock;

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::Tensor;

const SOURCE: &str = include_str!("metal/gemm.metal");

fn nax_version(ctx: &MetalContext) -> Result<MslVersion> {
    static VERSION: OnceLock<Result<Option<MslVersion>, String>> = OnceLock::new();
    let resolved = VERSION.get_or_init(|| {
        let probe = |v: MslVersion| ctx.pipeline("gemm_bf16_nt_nax", SOURCE, v).is_ok();
        Ok([MslVersion::V4_0, MslVersion::V4_1].into_iter().find(|v| probe(*v)))
    });
    resolved.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?.ok_or_else(|| {
        anyhow::anyhow!("the NAX GEMM compiles at no supported language version")
    })
}

pub fn gemm_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
) -> Result<()> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[0];
    ensure!(b.shape() == [n, k], "B shape {:?} != [{n}, {k}]", b.shape());
    // Output shape is arbitrary but must contain M*N contiguous elements.
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    let pipeline = ctx.pipeline("gemm_bf16_nt_nax", SOURCE, nax_version(ctx)?)?;
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), c.binding()],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(m)],
        Grid::Threadgroups {
            groups: (n.div_ceil(64), m.div_ceil(64), 1),
            threadgroup: (128, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/gemm.rs"]
mod tests;
