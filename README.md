# KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs

Implementation of the **KIVI algorithm** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) targeting Intel XPU hardware via SYCL/oneAPI. Replaces the original CUDA kernels with DPC++ and integrates with HuggingFace Transformers through a drop-in `KiviManager` class.

## Results

Benchmarked on Intel Iris Xe Graphics with GPT-2 (12 layers, 12 heads, head_dim=64), generating 200 tokens:

| Metric | FP32 Baseline | KIVI 2-bit |
|--------|:---:|:---:|
| **Speed** | 43.1 tok/s | 37.2 tok/s |
| **KV cache memory** | 1,386 KB | 378 KB |
| **Compression** | 1× | **3.67×** |
| **Token match vs FP32** | — | **100%** |
| **Overhead** | — | 16% |

The 100% token match is achieved through the CPU-residual architecture (see below), which eliminates floating-point rounding from unnecessary device transfers.

## How KIVI Works

The paper's key insight: keys and values have different outlier structures.

```
Keys:    outliers cluster in specific CHANNELS (dimensions)
         → quantize per-channel, grouping along the token axis

Values:  outliers cluster in specific TOKENS (positions)
         → quantize per-token, grouping along the channel axis
```

### Asymmetric 2-bit Quantization

```
scale = (max - min) / 3
zero  = min

Quantize:    q = clamp(round((x - zero) / scale), 0, 3)
Dequantize:  x ≈ q × scale + zero

Packing:     4 values → 1 byte
             byte = q₀ | (q₁ << 2) | (q₂ << 4) | (q₃ << 6)
```

### Residual Window

The most recent R tokens stay in full FP32 precision. Only older tokens beyond the window get quantized. When the residual buffer exceeds R tokens, the oldest G tokens are flushed to 2-bit storage:

```
Token stream: [t₀, t₁, ..., t_{n-R}, ..., tₙ]
               ├── Quantized (2-bit) ──┤├─ Residual (FP32) ─┤
```

## Architecture

```
┌──────────────────────────────────────────┐
│        Transformer Model (CPU)           │
│        HuggingFace GPT-2 etc.            │
└──────────────────┬───────────────────────┘
                   │ past_key_values (CPU tensors)
                   ▼
         ┌──────────────────┐
         │   KiviManager    │
         │   (benchmark.py) │
         └────────┬─────────┘
                  │
    ┌─────────────┴──────────────────┐
    │                                │
    ▼                                ▼
┌──────────────┐          ┌──────────────────┐
│  Residual    │          │  Quantized Store │
│  Buffer      │          │  (XPU Memory)    │
│  (FP32, CPU) │          │                  │
│              │          │ packed: uint8    │
│ Last R tokens│          │ scales: float32  │
│ No device    │          │ zeros:  float32  │
│ transfers    │          │                  │
└──────────────┘          └────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  SYCL Kernels     │
                         │  (kivi_sycl)      │
                         │                   │
                         │ quantize_keys()   │
                         │ quantize_values() │
                         │ dequantize_keys() │
                         │ dequantize_values()│
                         └───────────────────┘
```

### CPU-Residual Design

The critical performance optimization: **residual buffers live on CPU**, not XPU.

| Decode step type | What happens | XPU involved? |
|---|---|---|
| **Normal** (~97% of steps) | CPU `torch.cat` of cached history + residual | No |
| **Flush** (~3% of steps) | CPU→XPU transfer, SYCL quantize+dequantize, XPU→CPU transfer back | Yes |

This eliminates per-step XPU synchronization stalls that otherwise dominate latency on integrated GPUs. The dequantized history is cached on CPU and incrementally updated only when a flush occurs.

## Tensor Shapes

### Key Storage (Per-Channel)
```
Input:   [B, H, D, Seq]     float32  (keys transposed for per-channel quant)
Packed:  [B, H, D, Seq/4]   uint8    (4 values per byte)
Scales:  [B, H, D]          float32  (1 scale per channel)
Zeros:   [B, H, D]          float32  (1 zero-point per channel)
```

### Value Storage (Per-Token)
```
Input:   [B, H, Seq, D]             float32
Packed:  [B, H, Seq, D/4]           uint8    (4 values per byte)
Scales:  [B, H, Seq, D/group_size]  float32  (1 scale per group)
Zeros:   [B, H, Seq, D/group_size]  float32  (1 zero-point per group)
```

## Setup

### Prerequisites

- Intel GPU with oneAPI support (Arc, Iris Xe, Flex, Data Center)
- Intel oneAPI Base Toolkit 2025.0+
- Python 3.10+
- PyTorch with Intel Extension for PyTorch (IPEX)

### Install

```bash
# Activate oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Activate conda environment with IPEX
conda activate intel_xpu

# Build the SYCL extension
pip install . --no-build-isolation
```

### Verify

```bash
# Check SYCL devices are visible
sycl-ls

# Run kernel tests
python test/test_kivi.py
python test/test_quantize_simple.py
python test/test_attention_fix.py

# Run benchmark
python benchmark.py
```

## Usage

### Benchmark

```bash
python benchmark.py
```

Runs GPT-2 generation with both FP32 reference and KIVI quantization, reporting speed, compression ratio, and token-level quality comparison.

### Integrate with Your Model

```python
from benchmark import KiviManager

manager = KiviManager(
    num_layers=12,
    head_dim=64,
    device="xpu",
    group_size=32,       # G: elements per quantization group
    residual_length=64,  # R: full-precision window size
)

# Generation loop
for step in range(max_tokens):
    past = manager.get_full_cache()              # CPU tensors, zero-copy most steps
    out = model(input_ids, past_key_values=past, use_cache=True)
    manager.add_tokens(out.past_key_values)      # CPU-only, no XPU sync
    input_ids = out.logits[:, -1, :].argmax(-1).unsqueeze(1)
```

### Direct Kernel API

```python
import kivi_sycl

# Keys: per-channel quantization
kivi_sycl.quantize_keys(input, packed, scales, zeros, group_size)
kivi_sycl.dequantize_keys(packed, scales, zeros, output, group_size)

# Values: per-token quantization
kivi_sycl.quantize_values(input, packed, scales, zeros, head_dim, group_size)
kivi_sycl.dequantize_values(packed, scales, zeros, output, head_dim, group_size)
```

## Files

| File | Purpose |
|------|---------|
| `src/kivi_optimized.cpp` | SYCL kernels for 2-bit quantize/dequantize |
| `benchmark.py` | `KiviManager` class + GPT-2 benchmark with FP32 comparison |
| `setup.py` | Build config for the SYCL C++ extension |
| `test/test_kivi.py` | Comprehensive validation suite (15 checks) |
| `test/test_quantize_simple.py` | Deterministic known-value verification |
| `test/test_attention_fix.py` | End-to-end attention pipeline vs FP32 reference |

## Testing

Run the full test suite:

```bash
python test/test_kivi.py
python test/test_quantize_simple.py
python test/test_attention_fix.py
```

### test_kivi.py — 15 checks

| Test | What it verifies |
|------|------------------|
| Key round-trip | Per-channel quantize→dequantize error ≤ 0.5× scale |
| Value round-trip | Per-token quantize→dequantize error within tolerance |
| Known values | `[1,2,3,4]` → scale=1.0, zero=1.0, packed=228, lossless round-trip |
| Multi-batch/head | Correct behavior with B=2, H=4 |
| Constant input | No NaN/Inf for constant values, reconstructs exactly |

### test_quantize_simple.py

Deterministic bit-level verification with `[1,2,3,4]` across 4 channels:
- Confirms `packed = 0|(1<<2)|(2<<4)|(3<<6) = 228 = 11100100₂`
- Verifies scale=1.0, zero=1.0 across all channels
- Full round-trip: `[1,2,3,4] → quantize → dequantize → [1,2,3,4]` with 0.0 error

### test_attention_fix.py

End-to-end pipeline: quantize KV → dequantize → compute attention → compare with FP32:
- Key/value dequantization error ~1.0 (expected for 2-bit with random data)
- Cosine similarity between KIVI and FP32 attention output: **0.898** (> 0.85 threshold)
- No NaN/Inf in output

For production with larger models (7B+), R=128 is recommended per the paper.

## References

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — Zirui Liu et al., ICML 2024
- [Intel oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)

## License
BSD 3-Clause License
