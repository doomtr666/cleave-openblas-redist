# cleave-openblas-redist

A fork of [OpenMathLib/OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), pinned to `v0.3.33`,
whose only job is to build and publish a prebuilt OpenBLAS toolchain for
[cleave](https://github.com/doomtr666/cleave), the same way
[`cleave-llvm-redist`](https://github.com/doomtr666/cleave-llvm-redist) does for LLVM/MLIR.

## Why this exists

Real, measured GFLOP/s comparison on an AMD Ryzen 9 9700X (Zen 5), single
core, the exact matmul shape cleave's own `mnist-interop` kernel uses most
(`M=32, K=784, N=512`):

| | GFLOP/s | % of single-core roofline (353.34) |
|---|---|---|
| Cleave's own native codegen | ~96.9 | ~27% |
| MKL (measured via PyTorch) | ~113 | ~32% |
| **OpenBLAS** | **160.92** | **~45.5%** |

On a large square GEMM (a more favorable shape for BLAS packing/blocking),
OpenBLAS reaches **~225-230 GFLOP/s single-core (~64% of roofline)**, and
scales close to linearly with threads up to the machine's own core count
for large problems. MKL isn't an option for cleave (a proprietary,
non-redistributable binary, and cleave is explicitly meant to stay open
source) — OpenBLAS (BSD-3-Clause, genuinely open source, no separate
proprietary EULA the way AMD's own AOCL-BLAS binary distribution has) is
the real, viable path to closing a real gap in cleave's own generated
code, at least for the matmul-heavy hot path.

## Why v0.3.33, not the newer v0.3.34

Confirmed directly, not just read about: v0.3.34, built exactly this way
(Windows, clang-cl, `DYNAMIC_ARCH=ON`) on a real AMD Zen 5 machine with
AVX-512 exposed, segfaults on several real matrix sizes (500, 512, 900,
1000, 1024 — no simple pattern, e.g. not just "powers of two"). This is a
real, already-reported upstream regression —
[OpenMathLib/OpenBLAS#6013](https://github.com/OpenMathLib/OpenBLAS/issues/6013):
*"0.3.34 regression (Windows, clang-cl, DYNAMIC_ARCH): access violation in
dgemm_kernel_ZEN/HASWELL on AMD Zen 4/5 when AVX-512 is exposed; 0.3.33
fine."* The fix exists on `develop` but isn't in a tagged release yet, so
this repo pins the last known-good release instead. Bump forward once a
release actually containing the fix ships.

## Why clang-cl, not MinGW or plain MSVC

- **Plain MSVC (`cl.exe`) can't assemble OpenBLAS's own optimized kernels.**
  They're hand-written using GNU assembler syntax; `cl.exe` has no
  compatible assembler, so a pure-MSVC build silently falls back to the
  unoptimized "generic" C kernels — confirmed directly against OpenBLAS's
  own install docs, not assumed.
- **MinGW (OpenBLAS's own official Windows binaries use it) doesn't link
  cleanly into an ordinary Rust/MSVC target** — a different ABI/runtime
  than what `rustc`'s own `-pc-windows-msvc` target and its linker expect.
- **`clang-cl` is genuinely both**: a real GNU-compatible integrated
  assembler (so the hand-tuned kernels build as intended) *and*
  MSVC-ABI-compatible object files (so `link.exe`/Rust's own toolchain
  consume them with no bridging needed) — the same real trick AMD's own
  official AOCL-BLAS Windows binaries use (Clang 18, confirmed directly
  from their own release notes).

## Why `NOFORTRAN=1`

cleave only ever calls the BLAS entry points (`sgemm`/`dgemm`/...), never
LAPACK's own factorizations (SVD, eigendecomposition, ...). `NOFORTRAN=1`
skips LAPACK/Fortran entirely — a real, already-established OpenBLAS build
option (used by its own WASM/Emscripten and embedded/Android builds for
the identical reason), confirmed directly not to touch the optimized BLAS
Level 3 kernels at all (hand-written C/assembly/intrinsics in `kernel/`,
never Fortran). No `flang`/Conda dependency needed anywhere in this build.

## Why `DYNAMIC_ARCH=ON`

One binary, every CPU tier built in, real runtime dispatch by CPUID
feature detection — confirmed directly in `cpuid_x86.c` that a real AMD
Zen 5 chip with AVX-512 correctly routes to the `SKYLAKEX`/`COOPERLAKE`
kernel tier (not the AVX2-only `ZEN` one), and that this dispatch is by
actual instruction-set support, never a vendor-string check the way
Intel's own MKL dispatcher is documented to do. Fast on AMD *and* Intel,
not tuned for the build machine alone — the whole point of a
redistributable binary.

## Why `USE_OPENMP=1`

Shares the exact OpenMP runtime/thread pool cleave's own generated code
already links against, instead of racing it for cores with OpenBLAS's own
separate pthread-based threading model. Genuinely debrayable at runtime,
not a fixed build-time choice — confirmed directly:
`openblas_set_num_threads(1)` takes a real, separate single-threaded code
path internally (`level3.c`'s own driver checks the thread count first,
before ever touching OpenMP), not an OpenMP team degenerated to size one.
A caller can dial per-call between "use every core" and "run this one
matmul purely sequentially," which matters a lot for cleave's own
workloads: the real `mnist-interop` kernel's own dominant matmul shape
(`M=32`) measured *worse* at 16 threads than at 2 (thread synchronization
overhead dominating a thin matrix), while a large square GEMM scaled
almost linearly all the way to 16 — no single fixed thread count is right
for every shape cleave might generate.
