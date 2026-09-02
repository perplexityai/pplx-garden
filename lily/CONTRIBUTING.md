# Contributing

## Building

```sh
cargo build --release --locked
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

`rust-toolchain.toml` pins the compiler. Formatting is `rustfmt.toml`:
88 columns, edition-2024 style.

## Tests

```sh
cargo test --locked --lib                      # Rust/Metal kernel checks
cargo test --locked --test test_shader_compile # every kernel builds a pipeline
```

Those two need only a Metal device. The checkpoint-backed tokenizer and 35B
golden tests use `LILY_MODEL_DIR_35B` and are ignored when it is unset — see the
README.

Two things to know before trusting a green run:

- **A skipped test reports `ok`.** libtest captures the stdout of passing
  tests, so a target that found no checkpoint and returned early looks exactly
  like one that ran. The duration is the tell: `0.00s` means it did nothing.
- **The tests share one GPU.** Pass `--test-threads=1` for anything that loads
  a model.

## Adding a kernel

Every kernel is checked against a plain-Rust f32 reference in
`tests/support/cpu_ref.rs`, not against a previous version of itself. A new
kernel needs its reference in the same change.

If a kernel has a shape-dependent route, the route needs a boundary test: the
predicate at each edge, and numeric agreement on both sides of it. A route that
silently falls back is worse than one that errors, because a benchmark of the
fallback reads as a benchmark of the kernel.
