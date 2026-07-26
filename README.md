# KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs

Implementation of the **KIVI algorithm** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) targeting Intel XPU hardware via SYCL/oneAPI. Replaces the original CUDA kernels with DPC++ and provides a **plug-and-play** `kivi_sycl` module that works with **any** HuggingFace causal LM (GPT-2, LLaMA, Mistral, Phi, Qwen, Falcon, etc.).

```python
from kivi_sycl import generate
text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)
```

## Results

The packed-2-bit representation (uint8 payload + scales + zeros) is the resident CPU
store for flushed history — `KiviCache` no longer dequantizes-and-discards on flush, so
`get_stats()` reports **measured** resident bytes (`packed_bytes`, `residual_bytes`,
`kivi_bytes`, `fp_baseline_bytes`, `compression`), not a projected figure.

All measured 2026-07-25/26 on Intel Iris Xe Graphics, fp32, G=32, via
`get_stats()` — every byte count is read off live tensors, none are projected.

### Measured across models and context lengths

| Model | KV heads × head_dim | Layers | Context | Resident KV cache | Compression | Decode cost |
|---|---|---|---|---|---|---|
| gpt2 | 12 × 64 | 12 | 230 | 16.2 → 4.36 MB | 3.71× | −36% |
| TinyLlama-1.1B | 4 × 64 | 22 | 238 | 10.2 → 3.01 MB | 3.40× | −14% |
| TinyLlama-1.1B | 4 × 64 | 22 | 2k | 83.0 → 14.4 MB | 5.76× | −26% |
| Qwen2-0.5B | 2 × 64 | 24 | 4k | 96.7 → 14.7 MB | 6.58× | −27% |
| Qwen2.5-0.5B-Instruct | 2 × 64 | 24 | **8k** | 192.7 → **26.7 MB** | **7.22×** | −32% |

Short-context rows use R=64, long-context rows R=128. Cache sizes are summed over **all**
layers (`get_stats()` reports one layer; every layer holds identically shaped tensors, so
scaling by the layer count is exact). Throughput excludes prefill on both paths and counts
only tokens produced inside the timed region.

**Decode-cost figures are single runs and are noisy.** Repeat runs of an *identical*
configuration on this part have landed as far apart as −12% and −28%; the observed range
across all configurations is −12% to −28%, which is wider than any difference between the
configurations themselves. Treat every decode-cost number here as one sample from that
band, not as a measurement that distinguishes two settings. Resolving differences of this
size needs repeated runs with a median and a spread — not yet done. The memory columns do
not have this problem: they are deterministic byte counts, subject only to the sampling
phase noted below.

### Resident is not peak

Every figure above is **resident** — what the cache occupies between steps. KIVI's own
headline memory claim (2.6× for Llama-2-7B) is **peak**. They are different metrics, and
on this design they point in opposite directions.

A flush holds its tokens in three places until it is harvested: the FP residual, the
retained device-side kernel inputs, and the packed output. In steady-state decode that is
one group and is negligible. During bulk prefill it covers the prompt's entire grouped
part. Measured on TinyLlama-1.1B at ctx 1792, G=32, R=128, all 22 layers:

| | resident | peak | peak vs FP |
|---|---|---|---|
| FP baseline | 88.0 MB | — | 1.00× |
| KIVI | 15.8 MB (5.69×) | 159.0 MB | **0.55×** |
| KIVI, `prefill_chunk=256` | 15.8 MB (5.69×) | 103.8 MB | **0.85×** |

At its worst moment an unmitigated bulk prefill uses **1.8× the full-precision baseline**,
while resident sits 5.7× below it. `prefill_chunk=N` ingests a long prompt in slices of N,
harvesting between them, which bounds the transient to one slice; the packed store is
bit-identical either way, since slice boundaries are multiples of G and the quantization
groups do not move. It is off by default, so the table above is unaffected.

The residue at 0.85× is the full-precision reconstruction served to the model on every
step. Chunking cannot reach it — only fusing dequantization into attention can, which is
the same missing kernel that accounts for roughly half the decode cost. `get_stats()`
reports `peak_bytes`, `peak_at_tokens` and `peak_compression` alongside the resident
figures; peak is a measured lower bound (torch's own temporaries inside `torch.cat` are
not counted).

### Key-window flush policy

The paper quantizes the whole key residual when it reaches R and leaves the buffer empty,
so the key window averages R/2. This implementation defaults to draining one group at a
time, which averages ≈R — roughly twice the paper's full-precision key window.
`key_flush="reset"` selects the paper's policy (requires `R % G == 0`):

| `key_flush` | mean key window | compression (time-averaged) |
|---|---|---|
| `"drain"` (default) | 111.5 tok | 5.69× |
| `"reset"` (paper) | 63.5 tok | **6.07×** |

Measured at TinyLlama-1.1B, ctx 1792 + 256 generated, G=32, R=128. The key window halves
as predicted (63.5 against R/2 = 64), for **+6.7% compression**. No throughput column: see
the measurement caveats below — single runs on this part cannot resolve a difference this
size.

**Two things will produce wrong numbers if you skip them.**

*Compare with `mean_compression`, not `compression`.* The latter is a single instantaneous
reading, and the residual window sawtooths, so where a run happens to stop moves it by
±4%. At the exact context above both policies coincidentally report the same 5.58×, hiding
the whole 6.7% difference. This affects the model/context table further up too: those rows
are single readings, so each carries roughly ±4% of phase noise.

*Generate at least 2R tokens before trusting a `"reset"` measurement.* The key window
cycles over R steps. A run shorter than that samples only the low-occupancy start of the
ramp and reports compression that is too good — at `--gen 64` with R=128 the mean key
window reads 31.5 instead of its true 63.5, inflating compression to 6.37×. `"drain"`
cycles every G steps and is not affected.

### Compression does not depend on the model

The architectures above span 12 → 2 KV heads and 12 → 24 layers, but that variation does
not move the ratio. Per token, with `H` KV heads, head_dim `D`, group size `G`, and fp32
scales/zeros:

```
fp32 cache      = 2·H·D·4              = 8·H·D          bytes/token
2-bit payload   = 2·H·D·0.25           = 0.5·H·D
key   metadata  = H·D·2·4 / G          = 8·H·D/G        (per-channel, amortized over G tokens)
value metadata  = (D/G)·H·2·4          = 8·H·D/G        (per-token, one group per G channels)
                                       ─────────────────
quantized total = H·D·(0.5 + 16/G)
```

`H·D` cancels in the ratio, so the **ceiling is 8× for any architecture** at G=32. What
actually varies is how much of the cache is still in the FP residual window. With
`ρ = residual_tokens / total_tokens`:

> **compression = 8 / (1 + 7ρ)**

More generally, for a ceiling `C` (see *Raising the ceiling* below):
`compression = C / (1 + ρ·(C−1))`. The 8× form above is the G=32 / fp32-metadata case.

That single expression reproduces every measurement above to two decimal places:

| Run | H·D | fp32 B/tok | quantized B/tok | ρ | predicted | measured |
|---|---|---|---|---|---|---|
| gpt2 @230, R=64 | 768 | 6144 | 768 | 0.165 | 3.71× | **3.71×** |
| TinyLlama @238, R=64 | 256 | 2048 | 256 | 0.193 | 3.40× | **3.40×** |
| TinyLlama @2k, R=128 | 256 | 2048 | 256 | 0.055 | 5.76× | **5.76×** |
| Qwen2-0.5B @4k, R=128 | 128 | 1024 | 128 | 0.031 | 6.58× | **6.58×** |
| Qwen2.5-0.5B @8k, R=128 | 128 | 1024 | 128 | 0.015 | 7.22× | **7.22×** |

So the practical guidance is: **pick your context length and R, not your model.** A
larger model does not compress better — it just has a bigger cache to compress. The
8k run reaches 90% of the theoretical maximum because only 1.5% of its tokens are still
in the residual window.

Two independent checks that this is not curve-fitting: at *equal* conversation length
TinyLlama-1.1B and Qwen2.5-0.5B measure 5.17× and 5.36× despite a 2× difference in KV
heads; and the measured per-token packed cost (768 / 256 / 128 B) matches `H·D·(0.5+16/G)`
exactly for all three architectures.

### Raising the ceiling

The 8× ceiling is set by `G` and the metadata dtype, not the model. Metadata is currently
fp32 and, at G=32, costs exactly as much as the 2-bit payload itself:

| Group size | fp32 scales/zeros | fp16 scales/zeros |
|---|---|---|
| G=16 | 5.33× | 8.00× |
| **G=32** (default) | **8.00×** | 10.67× |
| G=64 | 10.67× | 12.80× |
| G=128 | 12.80× | 14.22× |

Storing scales/zeros in fp16 would lift the ceiling from 8× to 10.67× — a ~33%
improvement on the headline figure. The fp16 kernels already exist; see
`docs/status_and_roadmap.md` §6.2 for why this is currently blocked on this hardware.
Raising `G` costs quality (coarser quantization groups), so it is a real trade rather
than free headroom.

### Quality

WikiText-2 perplexity, gpt2, 10×512-token windows, G=32:

| Residual window | Perplexity | vs FP |
|---|---|---|
| FP32 baseline | 38.08 | — |
| KIVI, R=128 | 38.79 | **+1.87%** |
| KIVI, R=64 | 39.13 | **+2.76%** |

Halving the residual window roughly doubles the quality gap, in the direction the paper
predicts — the FP window is the mechanism that keeps 2-bit usable.

`examples/benchmark.py` also prints a greedy token-match rate, but read that number
carefully: greedy decoding diverges irreversibly, so it mostly encodes *where* the two
runs split rather than how much quality was lost.

For historical context, earlier versions of this README claimed 3.67×, computed
analytically while the flush path still dequantized-and-discarded — the actual resident
footprint then was ~1.0× (`docs/status_and_roadmap.md` §2.1).

### When KIVI helps

The savings scale with the cache, which grows with model size and context. At 2k on TinyLlama the KV cache is 83 MB against ~4.4 GB of fp32 weights, so the
68 MB saved is under 2% of resident memory — real but not decisive. At 8k on
Qwen2.5-0.5B the cache is 192.7 MB against 1885 MB of weights, and the 166 MB saved is
8.8% of total resident memory. It becomes decisive where the cache rivals the weights
(≥3B params at ≥8k, ~1 GB of fp32 KV). Note the decode cost grows with context too
(−26% at 2k → −32% at 8k): it is the price of reconstructing full precision on read, and
`docs/status_and_roadmap.md` §2.6 explains why it has a floor that only fused quantized
attention can remove.

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
│  Residual    │          │  Packed Store    │
│  Buffer      │          │  (CPU-resident)  │
│  (FP, CPU)   │          │                  │
│              │          │ packed: uint8    │
│ Last R tokens│          │ scales: fp32/16  │
│ No device    │          │ zeros:  fp32/16  │
│ transfers    │          │ dequant on read  │
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
| **Normal** (~97% of steps) | CPU-side unpack of the packed history + `torch.cat` with the residual | No |
| **Flush** (~3% of steps) | CPU→XPU transfer, SYCL quantize, packed/scales/zeros back to CPU | Yes |

This eliminates per-step XPU synchronization stalls that otherwise dominate latency on integrated GPUs.

Two things worth knowing about the flush path:

- **The packed representation is the resident store.** A flush quantizes only — it does not dequantize back. Full precision is reconstructed transiently on read, which is where the memory saving comes from and also where the decode overhead in the tables above comes from. `serving="cached"` keeps a persistent FP copy instead, trading the memory back for speed — but measured at 2k context that puts the resident cache *above* the FP baseline (0.89× compression), so it is rarely the right choice. `get_stats()` reports it honestly via `serving_copy_bytes`.
- **Flush is asynchronous by default.** The kernels are submitted and the call returns; the flushed tokens stay in the FP residual (still served correctly) until the next `add_tokens()` harvests them, so kernel and transfer time overlap the model's forward pass. Pass `async_flush=False` for synchronous harvest in the same call.

## Tensor Shapes

`T` below is the element dtype — `float32` or `float16` (bfloat16 is upcast to fp32 at
ingestion; there is no bf16 kernel). See `docs/tensor_spec_and_shapes.md` for the full
contract.

### Key Storage (Per-Channel)
```
Input:   [B, H, D, Seq]              T      (keys transposed for per-channel quant)
Packed:  [B, H, D, Seq/4]            uint8  (4 values per byte)
Scales:  [B, H, D, Seq/group_size]   T      (1 scale per G-token group, per channel)
Zeros:   [B, H, D, Seq/group_size]   T      (1 zero-point per group)
```

### Value Storage (Per-Token)
```
Input:   [B, H, Seq, D]             T
Packed:  [B, H, Seq, D/4]           uint8  (4 values per byte)
Scales:  [B, H, Seq, D/group_size]  T      (1 scale per G-channel group, per token)
Zeros:   [B, H, Seq, D/group_size]  T      (1 zero-point per group)
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
cache = KiviCache.from_model(
    model, group_size=32, residual_length=128,
    serving="rebuild",   # "rebuild" (default, memory-honest) or "cached" (faster,
                          # keeps an extra FP history copy resident — see below)
    async_flush=True,    # overlap flush kernels with the next forward pass
)

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
print(f"Packed (2-bit): {stats['packed_bytes']/1024:.0f} KB")
print(f"Residual (FP):  {stats['residual_bytes']/1024:.0f} KB")
print(f"KIVI resident:  {stats['kivi_bytes']/1024:.0f} KB vs FP baseline: {stats['fp_baseline_bytes']/1024:.0f} KB")
```

All fields are measured from live tensors (`numel() * element_size()`), not projected.
`serving="cached"` mode also reports `serving_copy_bytes` — the extra FP history copy it
keeps resident — folded into the `compression` denominator, so switching to `"cached"`
honestly shows less compression in exchange for speed.

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
# kernel launch, without ever materializing packed/scales/zeros. Useful when
# the packed representation itself is discarded by the caller — KiviCache's
# flush does NOT use this (packed/scales/zeros ARE its resident store); it
# calls quantize_keys_async/quantize_values_async instead (see below).
kivi_sycl_native.quant_dequant_roundtrip_keys(input, output, group_size)
kivi_sycl_native.quant_dequant_roundtrip_values(input, output, head_dim, group_size)

# Non-blocking variants of every op above (suffix _async) return a
# KiviEvent instead of blocking; call .wait() on it before reading output.
# KiviCache uses these by default (async_flush=True): a flush's quantize
# kernels are submitted during add_tokens() and harvested (packed store
# updated, residual trimmed) at the start of the *next* add_tokens() call,
# overlapping kernel + transfer time with the model's forward pass.
```

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `group_size` (G) | 32 | Quantization granularity. Smaller = better quality, more metadata |
| `residual_length` (R) | 128 | Recent FP32 tokens. Larger = better quality, less compression |

For GPT-2 (short sequences), `R=64` is sufficient. For 7B+ models with long contexts, `R=128` is recommended per the paper.

## Project Layout

The project uses a modular `src/`-layout: a pure-Python package (`src/kivi_sycl/`) and its native SYCL sources (`src/csrc/`) are kept fully separate, each broken down by single responsibility. See [`docs/architecture_and_state.md`](docs/architecture_and_state.md) for the full design rationale.

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
| `examples/chat.py` | Interactive terminal chat (ollama-style), Qwen2.5-0.5B-Instruct by default, with live compression stats |
| `examples/app.py` | Minimal one-liner demo |
| `scripts/build_wheel.sh` | Build pre-built wheel for distribution |
| `setup.py` | Package config with auto-detected SYCL paths |
| `pyproject.toml` | Modern Python packaging metadata |
| `docs/commands.md` | Command reference — run, test, benchmark, rebuild, gotchas |
| `docs/architecture_and_state.md` | Cache state machine, storage invariants, module design |
| `docs/status_and_roadmap.md` | What works, what's measured, what's still missing, known issues |
| `docs/sycl_kernel_interface.md` | Kernel / ops-layer contract, sync and async entry points |
| `docs/tensor_spec_and_shapes.md` | Exact shapes and dtypes across the Python↔SYCL boundary |
| `docs/dev-environment-setup.md` | From-scratch oneAPI + conda + build setup on this machine |
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
