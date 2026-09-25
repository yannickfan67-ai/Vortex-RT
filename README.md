# Vortex-RT

A performance-first AI inference runtime built directly on Vulkan Compute.

Vortex-RT is intentionally **not** a wrapper around llama.cpp. The project starts from a small Vulkan runtime and grows upward into tensor kernels, quantized GEMM, KV cache management, Transformer execution, GGUF loading, and an OpenAI-compatible serving layer.

## Goals

- Vulkan 1.3 compute backend for AMD, NVIDIA and Intel GPUs
- Low driver overhead: persistent pipelines, descriptor reuse, command reuse and explicit memory ownership
- Fast kernels: subgroup-aware tiled GEMM, fused normalization/activation, quantized matmul and attention
- FP32 / FP16 first, then INT8 and 4-bit weight formats
- Device-local weights and KV cache with staging only at model load / I/O boundaries
- No required CUDA, ROCm, DirectML or llama.cpp dependency
- CLI + embeddable C++ runtime; API server later

## Current milestone: M0

The first milestone establishes a small native Vulkan compute core:

- GPU selection with dedicated-compute queue preference
- Vulkan 1.3 device creation
- FP16 / INT8 / 8-bit / 16-bit capability discovery
- subgroup property discovery
- reusable buffer abstraction
- SPIR-V compute pipeline helper
- first compute kernel + benchmark harness

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
- Vulkan SDK / loader + headers
- `glslc` (normally shipped with the Vulkan SDK)

### Windows

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\vortexrt-info.exe
.\build\Release\vortexrt-bench.exe
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vortexrt-info
./build/vortexrt-bench
```

## Performance roadmap

1. Runtime bring-up and profiling
2. Device-local allocator + persistent staging arena
3. FP16 vector/tensor primitives
4. subgroup + shared-memory tiled GEMM
5. RMSNorm, RoPE, SiLU and fused SwiGLU
6. quantized Q8/Q4 matmul
7. Flash-style causal attention + paged KV cache
8. GGUF loader and Llama/Qwen-family execution
9. continuous batching and OpenAI-compatible API

## License

License selection is intentionally left open for the initial bring-up commit.
