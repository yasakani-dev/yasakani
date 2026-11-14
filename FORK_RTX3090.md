# RTX 3090 (sm_86) fork — unofficial

This is an **unofficial** fork of YASAKANI that adds an experimental build for the NVIDIA RTX 3090
(compute capability 8.6, sm_86). It is not affiliated with or endorsed by the YASAKANI authors.
The upstream code is licensed under Apache-2.0; see `LICENSE` and `NOTICE` (unchanged).
Based on upstream commit `91a5178`.

## Status: unverified

The sm_86 build compiles and links. **It has not been run on a GPU**: no load, no inference, no
accuracy or speed check has been done on an RTX 3090. Treat any output as untested.

## Build

```
cmake -S . -B build86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DYSKN_CUDA_ARCH=86
cmake --build build86
```

`YSKN_CUDA_ARCH` defaults to `120` (RTX 50 series), which builds the upstream code as before.

## What changed, and why

Only two PTX instructions in the source need sm_90 or newer, and both are compiled out below sm_90:

* `cp.async.bulk.prefetch` is a performance hint. Nothing depends on it.
* `griddepcontrol` is **not** just a hint: it is the synchronisation that programmatic dependent launch (PDL)
  relies on. It is removed only together with PDL itself, which is switched off at run time, so that kernels
  are launched plainly and stream order makes each one wait for the previous one.

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `YSKN_CUDA_ARCH` cache variable (default 120) sets `CMAKE_CUDA_ARCHITECTURES` and is passed to every source as a define. | Select the target; let host code know it. |
| `src/pdl.cuh` | `griddepcontrol.wait` / `.launch_dependents` only when `__CUDA_ARCH__ >= 900`; empty otherwise. | sm_90+ instructions (PDL). |
| `src/engine.cpp` | `sm90_features()` reads the GPU's compute capability once; `pdl_enabled()` is false below 9.0, so `launch_k` uses plain `<<<>>>` launches. | Without the attribute, stream order already makes each kernel wait for the previous one. |
| `src/q35_kernels.cu` | `cp.async.bulk.prefetch.L2.global` only when `__CUDA_ARCH__ >= 900`. | sm_90+ instruction (L2 prefetch, off by default). |
| `src/engine35.cpp` | Below 9.0, the L2 prefetch settings (`YSKN_L2PF`, `YSKN_L2PF2`, `set_opt` keys 25 and 27) are forced to 0. `load_bdraft` checks earlier CUDA errors before the SHEAD step. | No empty prefetch launches; the SHEAD fallback no longer hides earlier errors. |
| `src/loader.cpp` | `dev_copy` / `dev_zero` and the weight repack check CUDA return codes and throw. | Out-of-memory or launch failures during load used to pass silently (24 GB is tighter than 32 GB). |
| `src/c_api.cpp` | `yskn_open` throws if the GPU's compute capability is below the build target. | Clear error instead of "no kernel image" later. |

With `YSKN_CUDA_ARCH=120` the sm_120 device code (the cubins extracted from all 23 `.cu` objects)
was checked to be byte-for-byte the same as an unmodified build. Host-side changes only alter
behaviour when a CUDA call fails.

The `86` build keeps PDL and the L2 prefetch off on every GPU, also when it runs on a 9.0+ GPU, because
its device code has no `griddepcontrol`. It is meant for compute capability 8.x; on an RTX 50 series
card use the `120` build. Only `120` and `86` have been built; other values of `YSKN_CUDA_ARCH` are untried.

One case changes on any GPU: `yskn_open` with a device index that does not exist now fails, where the
upstream code silently used GPU 0.

Not changed: tuning values (workers, splits, grids), messages that mention sm_120, Python code.

## 日本語の要約

非公式のフォークです。RTX 3090 (sm_86) 向けに `-DYSKN_CUDA_ARCH=86` でビルドできるようにしました。
sm_90 以上の命令 2 種類 (PDL の griddepcontrol と L2 の一括先読み) を 8.6 では外し、実行時にも切っています。
加えて、ロード中の CUDA の失敗を黙って進めずに例外にしました。ビルドが通ることだけを確かめており、
3090 の実機での動作・精度・速度は未検証です。
