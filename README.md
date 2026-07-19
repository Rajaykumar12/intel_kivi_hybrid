# KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs

Implementation of the **KIVI algorithm** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) targeting Intel XPU hardware via SYCL/oneAPI. Replaces the original CUDA kernels with DPC++ and provides a **plug-and-play** `kivi_sycl` module that works with **any** HuggingFace causal LM (GPT-2, LLaMA, Mistral, Phi, Qwen, Falcon, etc.).

```python
from kivi_sycl import generate
text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)
```

## Results

Benchmarked on Intel Iris Xe Graphics with GPT-2 (12 layers, 12 heads, head_dim=64), generating 200 tokens:

| Metric | FP32 Baseline | KIVI 2-bit |
|--------|:---:|:---:|
| **Speed** | 16.3 tok/s | 14.4 tok/s |
| **KV cache memory** | 1,386 KB | 378 KB |
| **Compression** | 1× | **3.67×** |
| **Token match vs FP32** | — | **100%** |
| **Overhead** | — | 13% |

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
│   Any HuggingFace CausalLM (GPT-2,      │
│   LLaMA, Mistral, Phi, Qwen, etc.)      │
└──────────────────┬───────────────────────┘
                   │ past_key_values (CPU tensors)
                   ▼
         ┌──────────────────┐
         │    KiviCache     │
         │  (kivi_sycl)     │
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
                         │  (kivi_sycl._C)   │
                         │                   │
                         │ quantize_keys()   │
                         │ quantize_values() │
                         │ dequantize_keys() │
                         │ dequantize_values()│
                         └───────────────────┘
```

### Supported Models

`KiviCache.from_model()` auto-detects config from any HuggingFace `CausalLM`:

- **GPT-2** family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- **LLaMA** family (Llama-2, Llama-3, Code Llama)
- **Mistral** / **Mixtral**
- **Phi** (Phi-2, Phi-3)
- **Qwen** / **Qwen2**
- **Falcon**, **Gemma**, **GPT-J**, **GPT-NeoX**, **OPT**, **BLOOM**, **MPT**, **StableLM**
- Any model returning `past_key_values` as `tuple[tuple[Tensor, Tensor]]`

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

### Option 1: Install pre-built wheel (recommended for users)

No compiler needed — just an Intel GPU with drivers.

```bash
pip install kivi-sycl
```

Or install from a `.whl` file (e.g. from GitHub Releases):

```bash
pip install kivi_sycl-0.1.0-cp310-cp310-linux_x86_64.whl
```

**Requirements:**
- Intel GPU (Iris Xe, Arc, Flex, or Data Center)
- Intel GPU drivers installed ([install guide](https://dgpu-docs.intel.com/driver/installation.html))
- Python 3.10+

### Option 2: Build from source (for developers)

Needed only if you want to modify the SYCL kernels.

```bash
# Prerequisites: Intel oneAPI DPC++ compiler
source /opt/intel/oneapi/setvars.sh
conda activate intel_xpu

# Clone and install
git clone https://github.com/Rajaykumar12/intel_kivi_hybrid.git
cd intel_kivi_hybrid
pip install . --no-build-isolation
```

### Building a wheel (for maintainers)

```bash
source /opt/intel/oneapi/setvars.sh
bash scripts/build_wheel.sh
# Output: dist/kivi_sycl-0.1.0-cp310-cp310-linux_x86_64.whl
```

### Verify

```bash
# Check SYCL devices are visible
sycl-ls

# Run kernel tests
python tests/integration/test_kivi.py
python tests/integration/test_quantize_simple.py
python tests/integration/test_attention_fix.py

# Run benchmark
python examples/benchmark.py
```

## Usage

### Quick Start — One-liner Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kivi_sycl import generate

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# One line — works with any HuggingFace CausalLM
text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)
print(text)
```

### With Sampling Options

```python
text = generate(
    model, tokenizer,
    "The meaning of life is",
    max_new_tokens=300,
    do_sample=True,       # enable sampling
    temperature=0.8,      # creativity control
    top_k=50,             # top-k filtering
    verbose=True,         # print progress dots
)
```

### Custom Generation Loop (Full Control)

```python
from kivi_sycl import KiviCache
import torch

# Auto-configure from any HuggingFace model
cache = KiviCache.from_model(model, group_size=32, residual_length=128)

input_ids = tokenizer("Hello world", return_tensors="pt").input_ids

# Prefill
with torch.no_grad():
    out = model(input_ids, use_cache=True)
cache.add_tokens(out.past_key_values)
next_id = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

# Decode loop
for step in range(200):
    past = cache.get_full_cache()
    with torch.no_grad():
        out = model(next_id, past_key_values=past, use_cache=True)
    cache.add_tokens(out.past_key_values)
    next_id = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

# Reuse for another prompt
cache.reset()
```

### Check Memory Savings

```python
stats = cache.get_stats(layer=0)
print(f"Compression: {stats['compression']:.1f}x")
print(f"KIVI: {stats['kivi_bytes']/1024:.0f} KB vs FP32: {stats['fp32_bytes']/1024:.0f} KB")
```

### Use with Any Model

```python
# LLaMA
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
text = generate(model, tokenizer, "Explain quantum computing:", max_new_tokens=500)

# Mistral, Phi, Qwen, Falcon — same API, model config auto-detected
```

### Benchmark

```bash
python examples/benchmark.py                              # GPT-2 (default)
python examples/benchmark.py gpt2-medium                   # GPT-2 Medium
python examples/benchmark.py meta-llama/Llama-2-7b-hf      # LLaMA-2
python examples/benchmark.py --tokens 100 --G 32 --R 128   # custom params
```

### Direct Kernel API

The compiled SYCL extension is installed as the `kivi_sycl._C` submodule (not the top-level package, which is pure Python):

```python
from kivi_sycl import _C as kivi_sycl_native

# Keys: per-channel quantization
kivi_sycl_native.quantize_keys(input, packed, scales, zeros, group_size)
kivi_sycl_native.dequantize_keys(packed, scales, zeros, output, group_size)

# Values: per-token quantization
kivi_sycl_native.quantize_values(input, packed, scales, zeros, head_dim, group_size)
kivi_sycl_native.dequantize_values(packed, scales, zeros, output, head_dim, group_size)

# Fused round-trip: quantize to 2-bit fidelity and dequantize back in one
# kernel launch, without ever materializing packed/scales/zeros. Use this
# instead of a quantize_*+dequantize_* pair when the packed representation
# itself is discarded by the caller (this is what KiviCache's flush does).
kivi_sycl_native.quant_dequant_roundtrip_keys(input, output, group_size)
kivi_sycl_native.quant_dequant_roundtrip_values(input, output, head_dim, group_size)

# Non-blocking variants of every op above (suffix _async) return a
# KiviEvent instead of blocking; call .wait() on it before reading output.
```

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `group_size` (G) | 32 | Quantization granularity. Smaller = better quality, more metadata |
| `residual_length` (R) | 128 | Recent FP32 tokens. Larger = better quality, less compression |

For GPT-2 (short sequences), `R=64` is sufficient. For 7B+ models with long contexts, `R=128` is recommended per the paper.

## Project Layout

The project uses a modular `src/`-layout: a pure-Python package (`src/kivi_sycl/`) and its native SYCL sources (`src/csrc/`) are kept fully separate, each broken down by single responsibility. See [`docs/architecture.md`](docs/architecture.md) for the full design rationale.

```
src/
├── kivi_sycl/            # Python package (src-layout)
│   ├── __init__.py       # public API: KiviCache, generate
│   ├── backend/ipex.py   # IPEX/XPU detection
│   ├── config/           # HuggingFace model config auto-detection
│   ├── cache/manager.py  # KiviCache — the KV cache state manager
│   ├── generation/       # generate() loop + sampling strategies
│   └── extension/        # single import seam for the compiled kivi_sycl._C
└── csrc/                 # native C++/SYCL sources (not a Python package)
    ├── include/kivi/     # shared headers
    ├── kernels/           # SYCL quantize/dequantize kernels
    ├── ops/               # Torch tensor validation + kernel dispatch
    └── bindings/          # pybind11 module (kivi_sycl._C)
```

## Files

| File | Purpose |
|------|---------|
| `src/kivi_sycl/` | **Plug-and-play package**: `KiviCache` class + `generate()` one-liner (src-layout) |
| `src/csrc/` | SYCL kernels, ops, and pybind11 bindings for 2-bit quantize/dequantize |
| `examples/benchmark.py` | Benchmark script with FP32 comparison (uses `kivi_sycl`) |
| `examples/app.py` | Minimal one-liner demo |
| `scripts/build_wheel.sh` | Build pre-built wheel for distribution |
| `setup.py` | Package config with auto-detected SYCL paths |
| `pyproject.toml` | Modern Python packaging metadata |
| `docs/architecture.md` | Project layout / module design notes |
| `tests/integration/test_kivi.py` | Comprehensive validation suite (15 checks) |
| `tests/integration/test_quantize_simple.py` | Deterministic known-value verification |
| `tests/integration/test_attention_fix.py` | End-to-end attention pipeline vs FP32 reference |

## Testing

Run the full test suite:

```bash
python tests/integration/test_kivi.py
python tests/integration/test_quantize_simple.py
python tests/integration/test_attention_fix.py
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

## References

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — Zirui Liu et al., ICML 2024
- [Intel oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)

## License
BSD 3-Clause License
