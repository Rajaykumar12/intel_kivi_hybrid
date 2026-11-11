# KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs

Implementation of the **KIVI algorithm** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) targeting Intel XPU hardware via SYCL/oneAPI. Replaces the original CUDA kernels with DPC++ and provides a **plug-and-play** `kivi_sycl` module that works with **any** HuggingFace causal LM (GPT-2, LLaMA, Mistral, Phi, Qwen, Falcon, etc.).

```python
from kivi_sycl import enable_kivi
cache = enable_kivi(model)                 # native transformers Cache + "kivi" attention
out = model.generate(input_ids, past_key_values=cache, max_new_tokens=200)
print(cache.get_stats()["compression"])
```

Or with no Python at run time — export once, then run a self-contained C++ binary
([Python-free runtime](#python-free-runtime-kivi-run)):

```bash
python -m kivi_sycl.export --model Qwen/Qwen2.5-0.5B-Instruct --out artifacts/qwen25
kivi-run --model artifacts/qwen25 --chat
```

## Results

### Decode speed with the native integration (2026-11-08)

`enable_kivi` plugs KIVI into transformers as a real `Cache` plus a registered `"kivi"`
attention. Decode steps run a fused CPU kernel that attends directly over the packed 2-bit
history — the dequantized history is never materialised. Same memory, most of the decode
overhead gone. Three runs each; the percentage is KIVI vs the FP baseline measured in the
same process (the FP baseline itself varies ~20% run to run on this laptop):

| Model, context | Compression | Legacy loop (`get_full_cache`) | **Native + fused** |
|---|---|---|---|
| TinyLlama-1.1B, 1.9k + 64 | 6.99× | −1%, −18%, −28% (median **−18%**) | −6%, −2%, −2% (median **−2%**) |
| Qwen2.5-0.5B-Instruct, 8k + 32 | 9.25× | −41%, −30%, −35% (median **−35%**) | +6%, +18%, −10% (≈ **parity**) |

Per layer, the fused kernel is ~5× faster than dequantize-then-SDPA (Qwen2.5 shape, 8k:
5.8 → 1.0 ms). Quality is unchanged: gpt2 WikiText-2 perplexity 38.794 (fused) vs 38.793
(legacy). The native path's reference mode (`enable_kivi(model, fused=False)`) is
bit-identical to the legacy loop on gpt2, TinyLlama, Qwen2, Qwen2.5 and OPT.

The packed-2-bit representation (uint8 payload + scales + zeros) is the resident CPU
store for flushed history — `KiviCache` no longer dequantizes-and-discards on flush, so
`get_stats()` reports **measured** resident bytes (`packed_bytes`, `residual_bytes`,
`kivi_bytes`, `fp_baseline_bytes`, `compression`), not a projected figure.

All measured on Intel Iris Xe Graphics, fp32 models, G=32, via `get_stats()` — every
byte count is read off live tensors, none are projected.

### fp16 metadata (default since 2026-11-07)

Scales and zeros are now stored in fp16 for every model, which lifts the G=32 ceiling
from 8× to 10.67× (see *Raising the ceiling*). Same inputs, A/B on `--metadata`:

| Model | Context | fp32 metadata | **fp16 metadata** | Formula (fp16) |
|---|---|---|---|---|
| TinyLlama-1.1B | 1.9k + 32 | 83.0 → 14.4 MB, 5.76× | 83.0 → **11.9 MB, 6.95×** | 6.95× |
| Qwen2.5-0.5B-Instruct | 8k + 32 | 192.7 → 26.7 MB, 7.22× | 192.7 → **20.8 MB, 9.28×** | 9.28× |

The fp32-metadata column reproduces the July measurements below exactly. Pass
`metadata_dtype=torch.float32` (or `--metadata fp32` to the eval scripts) for the old
behaviour.

### Measured across models and context lengths (fp32 metadata, 2026-07-25/26)

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

The ceiling is set by `G` and the metadata dtype, not the model:
`C = 8 / (0.5 + 4m/G)` against an fp32 baseline, with m the metadata width in bytes
(4 for fp32, 2 for fp16). At G=32, fp32 metadata costs exactly as much as the 2-bit
payload itself:

| Group size | fp32 scales/zeros | fp16 scales/zeros |
|---|---|---|
| G=16 | 5.33× | 8.00× |
| **G=32** (default) | **8.00×** | 10.67× |
| G=64 | 10.67× | 12.80× |
| G=128 | 12.80× | 14.22× |

fp16 metadata is now the default (the table at the top). It was previously blocked on
this hardware because torch's XPU dtype-cast kernel fails on Iris Xe
(`docs/status_and_roadmap.md` §6.2); the quantize kernels now narrow the scale and zero to
fp16 themselves, so fp32 data with fp16 metadata never goes through that cast. For fp16
*models* the metadata was already fp16, and their ceiling against an fp16 baseline is
5.33×. Raising `G` costs quality (coarser quantization groups), so it is a real trade
rather than free headroom.

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
┌──────────────────────────────────────────────────────────┐
│ Transformer model (CPU) — any HuggingFace CausalLM        │
└───────────────┬──────────────────────────────────────────┘
                │ enable_kivi(model)            KiviCache.add_tokens /
                │ (transformers Cache +         get_full_cache (legacy loop)
                │  "kivi" / "kivi_op" attention)        │
                ▼                                       ▼
┌──────────────────────────────────────────────────────────┐
│ kivi_sycl (Python, thin)   hf/  cache/manager.py          │
│   adapts transformers objects; holds no cache state       │
└───────────────┬──────────────────────────────────────────┘
                │ _C.Engine  /  torch.ops.kivi.*
                ▼
┌──────────────────────────────────────────────────────────┐
│ libkivi_core (C++, no Python)              src/csrc/      │
│   kivi::Engine   residual · flush/harvest · reserve ·     │
│                  per-layer G · stats · attend()           │
│   LayerArena     packed 2-bit store (USM host on XPU)     │
│   ops            torch.ops.kivi.{quantize_keys,           │
│                  quantize_values, fused_decode_attention, │
│                  layer_attention}                         │
│   SYCL kernels   quantize on the iGPU, zero-copy          │
│   CPU kernel     fused decode attention (AVX2/FMA)        │
└──────────────────────────────────────────────────────────┘
```

Since Phase 5 all cache state and work lives in the C++ engine; the Python package only
adapts it to transformers. The same `libkivi_core.so` builds without Python (CMake), and the
Python-free runtime is built on it:

```
 offline, once (Python)                         run time (no Python)
┌───────────────────────────────┐   artifact   ┌──────────────────────────────────────┐
│ python -m kivi_sycl.export     │   directory  │ kivi-run (C++)            src/runtime/│
│  op-mode model → torch.export  │ ──────────►  │  AOTInductor package loader          │
│  → AOTInductor package (code)  │ model.pt2    │  weights: safetensors mmap → fp32    │
│  + weights manifest, configs,  │ *.safetensors│  kivi::Engine registered as id 0     │
│  effective tokenizer, verified │ *.json       │  tokenizer.json BPE · chat templates │
│  chat template pieces          │              │  sampler · streaming · stats         │
└───────────────────────────────┘              └──────────────────┬───────────────────┘
                                                                   │ torch.ops.kivi.layer_attention
                                                                   ▼
                                                        libkivi_core (as above)
```
With `device="cpu"` the engine uses a reference quantize instead of the SYCL kernels, so KIVI
also runs on machines without an Intel GPU (slowly).

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
git clone https://github.com/Rajaykumar12/intel_kivi_hybrid.git
cd intel_kivi_hybrid
bash scripts/dev_build.sh        # clean in-place build
bash scripts/dev_test.sh         # test suite
```

The compiler must be compatible with the SYCL runtime bundled in your torch XPU wheel.
`scripts/dev_build.sh` sources a specific oneAPI compiler (`KIVI_ICPX_VER`, default 2026.1)
instead of `setvars.sh`, which resolves to the newest installed oneAPI and can break
`import torch` (`undefined symbol: urKernelSetArgLocal`). `KIVI_ENV` selects the conda env.

**Ahead-of-time GPU code (optional):** `KIVI_AOT_DEVICE=adl-p bash scripts/dev_build.sh`
compiles native code for Alder Lake Iris Xe in addition to portable SPIR-V. Any `ocloc
-device` value works (`tgllp`, `dg2`, a family such as `xe-lp`, or a PCI id). Measured on a
cold driver cache, it removes the first-use JIT of the kernels (quantize 73 → 0.6 ms,
dequantize 33 → 0.3 ms per process); with a warm driver cache JIT is already ~1 ms.

`bash scripts/check_warnings.sh` audits the C++/SYCL sources with `-Wall -Wextra`
(the regular build uses `-w` to keep torch's header warnings out of the log).

### Building a wheel (for maintainers)

```bash
bash scripts/build_wheel.sh                         # portable
KIVI_AOT_DEVICE=adl-p bash scripts/build_wheel.sh   # + native Iris Xe code
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

### Native transformers integration (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kivi_sycl import enable_kivi

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok("Once upon a time", return_tensors="pt").input_ids

cache = enable_kivi(model, max_tokens=ids.shape[1] + 200)   # max_tokens is optional
out = model.generate(ids, past_key_values=cache, max_new_tokens=200)
print(tok.decode(out[0]), cache.get_stats()["compression"])
```

`enable_kivi` switches the model to the `"kivi"` attention and returns a fresh cache — use one
per sequence. It accepts the same options as `KiviCache` (`group_size`, `residual_length`,
`metadata_dtype`, `zero_copy`, …) plus `fused` (default `True`; `False` selects the
dequantize-then-SDPA reference path). Beam search and cache cropping are not supported yet.

### Compiled decode (`torch.compile`, op mode)

`enable_kivi_op` turns each attention layer's KIVI work into a single
`torch.ops.kivi.layer_attention` call, so `torch.compile` captures a whole decode step with
**no graph breaks** (the default path breaks twice per layer). Pass `position_ids` and the
cache's `model_kwargs` on every call:

```python
from kivi_sycl.hf.compiled import enable_kivi_op

cache = enable_kivi_op(model, max_tokens=ids.shape[1] + 200)
step = torch.compile(model.forward, dynamic=False)
out = model(ids, past_key_values=cache, position_ids=torch.arange(ids.shape[1])[None],
            **cache.model_kwargs)
pos = ids.shape[1]
nxt = out.logits[:, -1].argmax(-1, keepdim=True)
for _ in range(200):
    out = step(input_ids=nxt, past_key_values=cache, position_ids=torch.tensor([[pos]]),
               **cache.model_kwargs)
    nxt, pos = out.logits[:, -1].argmax(-1, keepdim=True), pos + 1
```

Measured (2026-11-10): compiled decode **1.20×** on Qwen2.5-0.5B @ 2k and **1.11×** on
TinyLlama-1.1B @ 1.9k over eager, greedy tokens unchanged. Batch size 1, no padding masks.

### Python-free runtime (`kivi-run`)

`kivi-run` runs an exported model with no Python in the process (checked: `/proc/<pid>/maps`
maps no `libpython`). Python is used once, offline, to export:

```bash
# 1. export (Python env): writes artifacts/qwen25/ — ~30 s, a few GB of RAM while compiling
python -m kivi_sycl.export --model Qwen/Qwen2.5-0.5B-Instruct --out artifacts/qwen25 \
    --max-tokens 2048 --group-size 32 --residual 128

# 2. build the runtime (CMake; links LibTorch from the torch wheel, not Python)
bash scripts/build_native.sh                 # build-native/kivi-run + ctest

# 3. run
build-native/kivi-run --model artifacts/qwen25 --chat               # stdin chat, /reset, /quit
build-native/kivi-run --model artifacts/qwen25 --chat --prompt "Hi" --system "Be brief."
build-native/kivi-run --model artifacts/qwen25 --prompt "Once upon a time" --max-new-tokens 64
build-native/kivi-run --model artifacts/qwen25 --prompt "..." --temperature 0.8 --top-p 0.95 --seed 7
build-native/kivi-run --model artifacts/qwen25 --ids 151644,872,198 --max-new-tokens 16 --stats
```

The artifact directory holds a code-only AOTInductor package (`model.pt2`, a few MB), the
model's own safetensors weights plus a manifest mapping every package constant to them, the
engine configuration (`kivi_config.json`), the tokenizer pipeline transformers actually runs,
and `generation.json` (chat template pieces and the model's default sampling).

| | TinyLlama-1.1B-Chat | Qwen2.5-0.5B-Instruct |
|---|---|---|
| export | 46 s, 2.1 GB dir | 26 s, 0.96 GB dir |
| load (warm page cache) | 0.4–0.7 s | 0.7–1.1 s |
| peak RSS | 4.7 GB | 2.4 GB |
| decode, short prompt¹ | 7.2 tok/s | 14.1 tok/s |
| greedy tokens vs Python (200) | identical | identical |

¹ Busy desktop, not the idle-machine protocol; the comparison with the Python paths is
pending (`docs/optimization_roadmap/phase5_python_free_runtime.md` §4.6).

**What it implements, and how it is checked**
- *Tokenizer:* a `tokenizer.json` reader + BPE in C++ (Llama/SentencePiece-style Metaspace with
  byte fallback, and byte-level GPT-2/Qwen with a PCRE2 regex split and ICU NFC). The exporter
  saves `backend_tokenizer` — the pipeline transformers runs, which can differ from the file a
  model repo ships (transformers 5 swaps TinyLlama's normalizer for a Metaspace pre-tokenizer).
  Conformance: ~2.4k cases per model (WikiText-2, Unicode stress, code, random-id decodes)
  identical to transformers, encode and decode.
- *Chat:* no Jinja in C++. The exporter derives each template's literal pieces (role
  prefix/suffix, system block, default system prompt, generation prompt) by rendering sentinel
  conversations, verifies the reconstruction byte-for-byte against the template, and refuses
  templates it cannot reproduce (e.g. ones that `trim` content). Multi-turn chat keeps the KV
  cache and feeds only each turn's new tokens; replies match HF `generate()` turn by turn.
- *Sampling:* greedy, temperature, top-k, top-p, repetition penalty (HF order), seeded and
  reproducible. `--chat` uses the model's `generation_config` defaults unless a flag is given.
- *Memory:* weights are memory-mapped; tensors converted to fp32 release their bf16 pages at
  once, so peak RSS equals the steady state.

**Relocatable package.** `bash scripts/package_runtime.sh` produces
`dist/kivi-runtime-<ver>-linux-x86_64/` (+ `.tar.zst`): `bin/kivi-run` with an
`$ORIGIN/../lib` RPATH and `lib/` holding LibTorch, the SYCL runtime, oneMKL, ICU and PCRE2
copied from the build environment (1.9 GB unpacked, 808 MB compressed — `libtorch_xpu` alone is
874 MB), with `THIRD_PARTY_NOTICES.txt` and license texts. The target needs glibc ≥ 2.38 and,
for `--device xpu`, the Intel GPU driver (Level Zero). `--check <artifacts> <golden>`
extracts the tarball and runs it under `env -i` — no conda, Python or oneAPI — and verifies
every mapped library comes from the package or the OS. Several bundled Intel runtime
libraries are under Intel's EULA / Simplified Software License: check their redistribution
terms before shipping the tarball.

**Limits.** Batch size 1; the model runs in fp32 on the CPU (the KV engine uses the iGPU);
the context limit is fixed at export (`--max-tokens`); an artifact runs only with a runtime
built on the same torch release (kivi-run warns on a mismatch); tested on the Llama
(TinyLlama) and Qwen2 families.

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
| `group_size` (G) | 32 | Quantization granularity. Smaller = better quality, more metadata. Also accepts a list with one G per layer (see below) |
| `residual_length` (R) | 128 | Recent FP32 tokens. Larger = better quality, less compression |
| `metadata_dtype` | `"auto"` (fp16) | Scale/zero dtype. fp16 lifts the G=32 ceiling to 10.67×; `torch.float32` restores 8× (fp32 models only) |
| `zero_copy` | `"auto"` | On XPU, keep the packed store in USM host memory that the kernels write directly — no host↔device copies. `False` stages through XPU tensors |
| `cache.reserve(n)` | — | Optional: size the packed store once for an `n`-token sequence (`generate()` calls it), so it never grows and carries no spare capacity |

**Per-layer group size.** Which layers get coarse groups matters far more than how many:
on gpt2 (WikiText-2, 10×512, R=128), coarsening the middle layers to G=64 costs ~+5% perplexity,
while coarsening the first two layers is nearly free:

| `group_size` | Ceiling (vs fp32) | Perplexity Δ vs FP |
|---|---|---|
| `32` (default) | 10.67× | +1.88% |
| `[64, 64] + [32] * 10` | 10.97× | +1.89% |
| `[64, 64] + [32] * 8 + [64, 64]` | 11.29× | +2.02% |
| `[16, 16] + [64] * 10` (wrong layers coarsened) | 11.64× | +5.14% |
| `64` | 12.80× | +5.23% |

That is one model on ~5k tokens — measure your own model with
`examples/eval_perplexity.py --G 64,64,32,...` before relying on a schedule.

For GPT-2 (short sequences), `R=64` is sufficient. For 7B+ models with long contexts, `R=128` is recommended per the paper.

## Project Layout

```
src/
├── kivi_sycl/              # Python package (thin front end)
│   ├── __init__.py         # public API: KiviCache, enable_kivi, generate
│   ├── cache/manager.py    # KiviCache — legacy loop API over the C++ engine
│   ├── hf/                 # transformers integration: Cache, "kivi" attention, op mode
│   ├── config/             # HuggingFace model config auto-detection
│   ├── generation/         # generate() loop + sampling strategies
│   ├── export/             # exporter: torch.export → AOTInductor, artifacts, chat templates
│   └── extension/          # import seam for the compiled kivi_sycl._C
└── csrc/                   # native C++/SYCL (builds into _C and libkivi_core.so)
    ├── engine/             # kivi::Engine, LayerArena, CPU dequant, engine registry
    ├── library/            # TORCH_LIBRARY(kivi): torch.ops.kivi.*
    ├── cpu/                # fused decode attention (AVX2/FMA + generic)
    ├── kernels/            # SYCL quantize/dequantize kernels
    ├── ops/                # tensor validation + kernel dispatch
    ├── include/kivi/       # shared headers (queue, USM, layout, events)
    └── bindings/           # pybind11 module (kivi_sycl._C)
src/runtime/                # kivi-run: artifacts + AOTI loader, tokenizer/, chat, sampler
third_party/nlohmann/       # vendored nlohmann/json 3.11.3 (MIT)
tools/                      # golden generators for the C++ tests (Python, offline)
CMakeLists.txt              # Python-free libkivi_core.so, kivi-run, C++ tests
tests/{unit,integration,native}/
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
| `src/kivi_sycl/export/` | Exporter (`python -m kivi_sycl.export`): AOTInductor package, artifact directory, chat-template derivation |
| `src/runtime/` | `kivi-run`, the Python-free C++ runtime (model loader, tokenizer, chat, sampler) |
| `scripts/build_native.sh` | CMake build of `libkivi_core.so`, `kivi-run` and the C++ tests, then ctest |
| `scripts/package_runtime.sh` | Relocatable `kivi-run` package + tarball, with a clean-environment check |
| `tools/runtime_golden.py`, `tools/tokenizer_golden.py` | Python reference outputs (greedy tokens, chat replies, tokenizer corpus) the C++ tests compare against |
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

```bash
bash scripts/dev_test.sh            # pytest: unit (CPU) + integration (XPU)
bash scripts/build_native.sh        # C++ build of libkivi_core.so + ctest (no Python)
bash scripts/check_warnings.sh      # -Wall -Wextra audit of src/csrc
python examples/verify_hf_equivalence.py   # 5 real models: legacy vs native path bit-identical
```

- `tests/unit/` run on the CPU (the engine's reference quantize); integration tests need an
  Intel GPU and skip cleanly without one.
- `tests/native/` are C++ programs: the arena, the fused attention, the dispatcher ops, and an
  assertion that no libpython is loaded.
- Runtime goldens (ctest, registered when the files exist — artifacts are generated, not
  checked in): `kivi_run_golden_<model>` (200 greedy tokens identical to Python eager op mode),
  `..._chunked` (same, prefill in 4-token chunks — the path later chat turns take),
  `kivi_run_chat_<model>` (3-turn chat replies identical to HF `generate()`),
  `tokenizer_<model>` (tokenizer + chat-template conformance). Generate them with:

  ```bash
  python tools/runtime_golden.py --artifacts artifacts/qwen25 --model Qwen/Qwen2.5-0.5B-Instruct \
      --out build-native/golden/qwen25_run.json
  python tools/runtime_golden.py --artifacts artifacts/qwen25 --model Qwen/Qwen2.5-0.5B-Instruct \
      --chat-turn "Capital of France? One word." --chat-turn "And of Japan?" \
      --reply-tokens 12 --out build-native/golden/qwen25_chat.json
  python tools/tokenizer_golden.py --artifacts artifacts/qwen25 \
      --out build-native/golden/qwen25_tokenizer.json
  bash scripts/build_native.sh       # re-configures and picks them up
  ```
- Equivalence chain kept in the suite: legacy loop ≡ HF reference path (bit-exact);
  fused kernel ≈ reference (1e-4); op mode ≡ default fused path (bit-exact); compiled op mode
  ≈ eager (same greedy tokens).

## References

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — Zirui Liu et al., ICML 2024
- [Intel oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)

## License
BSD 3-Clause License
