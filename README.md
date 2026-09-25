# Vortex-RT

A performance-first AI inference runtime built directly on Vulkan Compute.

Vortex-RT is intentionally **not** a wrapper around llama.cpp. The project starts from a compact Vulkan runtime and grows upward into tensor kernels, quantized GEMM, KV-cache management, Transformer execution, GGUF loading and a serving layer.

## Current state

The runtime foundation is now usable rather than just a compile-only bring-up:

- Vulkan 1.3 compute backend with dedicated-compute queue preference
- Explicit device selection by index/name with `--device` or `VORTEXRT_DEVICE`
- FP16 / INT8 / 8-bit / 16-bit feature discovery
- subgroup and memory-capability reporting
- pooled device-local memory suballocation
- persistent host-visible staging arena
- reusable transfer command buffer + fence
- descriptor reuse and cached compute command recordings
- SPIR-V compute pipeline helper
- vector-add correctness/performance benchmark
- GGUF v2/v3 metadata + tensor-directory parser
- Vulkan-native Q8_0 matvec smoke kernel with CPU-reference validation
- explicit transfer/compute memory dependencies for cross-driver correctness
- Gemma 3 270M Q8_0 GGUF validation in CI
- Lavapipe CPU-Vulkan smoke tests in GitHub Actions

## Goals

- Vulkan 1.3 compute backend for AMD, NVIDIA and Intel GPUs
- Low driver overhead: persistent pipelines, descriptor reuse, command reuse and explicit memory ownership
- Fast kernels: subgroup-aware tiled GEMM, fused normalization/activation, quantized matmul and attention
- FP32 / FP16 first, then INT8 and 4-bit weight formats
- Device-local weights and KV cache with staging only at model-load / I/O boundaries
- No required CUDA, ROCm, DirectML or llama.cpp dependency
- CLI + embeddable C++ runtime; API server later

## Planned execution stack

```text
model / tokenizer / sampler
          |
Transformer executor
          |
graph + tensor scheduler
          |
fused AI kernels
          |
Vortex Vulkan runtime
          |
Vulkan 1.3
          |
AMD / NVIDIA / Intel
```

## Build prerequisites

- CMake 3.24+
- C++20 compiler
- Vulkan 1.3 loader + development headers
- `glslc`

### Windows

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\vortexrt-info.exe
.\build\Release\vortexrt-bench.exe
.\build\Release\vortexrt-gguf-info.exe model.gguf
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vortexrt-info
./build/vortexrt-bench
./build/vortexrt-gguf-info model.gguf
```

Run the small Vulkan smoke suite with:

```bash
ctest --test-dir build --output-on-failure
```

## Device selection

Automatic selection prefers discrete GPUs, but multi-GPU systems can override it:

```bash
./build/vortexrt-info --device 1
./build/vortexrt-bench --device "RTX 5070"
VORTEXRT_DEVICE="RX 9070" ./build/vortexrt-info
```

The numeric selector is the Vulkan physical-device index reported by `vortexrt-info`.

## Benchmark controls

The default benchmark is intentionally large. For smoke testing or software Vulkan:

```bash
./build/vortexrt-bench --elements 65536 --repetitions 2
```

The reported bandwidth is **logical kernel bandwidth** (bytes requested by the vector-add kernel), not a claim of raw DRAM bandwidth.

## GGUF probe

`vortexrt-gguf-info` validates the GGUF header, metadata, tensor directory, alignment and supported tensor payload bounds without loading the full model into GPU memory.

```bash
./build/vortexrt-gguf-info gemma-3-270m-Q8_0.gguf --expect-arch gemma3
```

The CI path downloads the published Gemma 3 270M Q8_0 GGUF and checks that its tokenizer metadata and tensor table are readable.

## Performance roadmap

1. Runtime bring-up and profiling — **done**
2. Device-local allocator + persistent staging arena — **done**
3. GGUF metadata/tensor parsing + real-model CI — **done**
4. FP16 vector/tensor primitives
5. subgroup + shared-memory tiled GEMM
6. RMSNorm, RoPE, SiLU and fused SwiGLU
7. Q8_0 matvec bring-up — **done**; optimize/deploy GGUF-backed quantized GEMM next
8. Transformer execution + KV cache
9. Q4/K-quants and fused attention
10. continuous batching and OpenAI-compatible API

## License

License selection is still open.
