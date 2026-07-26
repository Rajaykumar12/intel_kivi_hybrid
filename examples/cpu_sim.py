"""End-to-end KIVI pipeline test on CPU — no GPU, XPU, oneAPI, or compiled
extension required.

Replaces the native SYCL kernels with an exact torch reference of the
asymmetric 2-bit round-trip and runs the real generate()/KiviCache
pipeline against a real HuggingFace model, comparing with vanilla greedy
generation.

Two modes:
  default     — real 2-bit rounding. Verifies the whole pipeline and lets
                you judge output quality. Some token divergence from the
                FP32 baseline is EXPECTED (2-bit is lossy); the output
                must stay coherent, not turn to gibberish.
  --identity  — the "quantization" is a lossless copy, so the pipeline
                must reproduce the FP32 baseline token-for-token. This is
                the strongest CPU check of the cache bookkeeping: any
                dropped, reordered, or stale token breaks the 100% match.

Usage:
  python examples/cpu_sim.py                    # gpt2, 2-bit, 120 tokens
  python examples/cpu_sim.py --identity         # must print 100% match
  python examples/cpu_sim.py distilgpt2 --tokens 80 --G 16 --R 32
"""

import argparse
import os
import sys
import types

import torch

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)


class _FakeEvent:
    def wait(self):
        pass


def _quantize_2bit(inp, packed_out, scales_out, zeros_out, group_size):
    """Exact torch mirror of the SYCL quantize kernels, including the
    LSB-first packing (byte = q0 | q1<<2 | q2<<4 | q3<<6). Both kernels
    reduce over contiguous runs of `group_size` elements, so a flat
    [-1, group_size] view reproduces the grouping for keys ([B,H,D,L],
    groups along tokens) and values ([B,H,L,D], groups along channels)
    alike."""
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0).to(torch.uint8)
    qb = q.reshape(-1, group_size // 4, 4)
    byte = (qb[..., 0] | (qb[..., 1] << 2) | (qb[..., 2] << 4)
            | (qb[..., 3] << 6))
    packed_out.copy_(byte.reshape(packed_out.shape))
    scales_out.copy_(scale.reshape(scales_out.shape))
    zeros_out.copy_(mn.reshape(zeros_out.shape))


def _install_identity_shortcut(KiviCache):
    """Make the flushed history round-trip losslessly.

    2-bit storage cannot represent its input exactly, so identity mode
    cannot be expressed as a lossless *codec* — it has to bypass the packed
    store. The flushed FP chunks are stashed per layer on the way in and
    replayed on the way out, leaving every other part of the manager (flush
    sizing, residual trimming, async harvest, cache-view invalidation) doing
    its real work. Any dropped, reordered, or stale token still shows up as
    a token mismatch; only the quantization noise is removed."""
    submit_flush = KiviCache._submit_flush

    def _submit_flush(self, layer_idx, flush_len):
        shadow = self.__dict__.setdefault("_identity_shadow", {})
        k, v = shadow.setdefault(layer_idx, ([], []))
        k.append(self.res_keys[layer_idx][:, :, :flush_len, :].clone())
        v.append(self.res_values[layer_idx][:, :, :flush_len, :].clone())
        submit_flush(self, layer_idx, flush_len)

    def _serve_layer(self, layer_idx):
        # Chunks are stashed at *submit* time, but a chunk only becomes
        # history at *harvest* time — until then its tokens are still being
        # served from the FP residual. Truncating to the manager's own
        # quantized-token count keeps an in-flight async flush from being
        # served twice.
        n = self._quant_token_count[layer_idx]
        res_k, res_v = self.res_keys[layer_idx], self.res_values[layer_idx]
        n_res = res_k.shape[2] if res_k is not None else 0
        if n == 0:
            return (res_k, res_v) if n_res else (None, None)
        k, v = self.__dict__["_identity_shadow"][layer_idx]
        hist_k = torch.cat(k, dim=2)[:, :, :n, :]
        hist_v = torch.cat(v, dim=2)[:, :, :n, :]
        if not n_res:
            return hist_k, hist_v
        return (torch.cat([hist_k, res_k], dim=2),
                torch.cat([hist_v, res_v], dim=2))

    KiviCache._submit_flush = _submit_flush
    KiviCache._serve_layer = _serve_layer


def install_fake_native(identity):
    fake = types.ModuleType("kivi_sycl._C")

    def quantize_keys_async(inp, packed, scales, zeros, group_size):
        _quantize_2bit(inp, packed, scales, zeros, group_size)
        return _FakeEvent()

    def quantize_values_async(inp, packed, scales, zeros, head_dim, group_size):
        _quantize_2bit(inp, packed, scales, zeros, group_size)
        return _FakeEvent()

    fake.quantize_keys_async = quantize_keys_async
    fake.quantize_values_async = quantize_values_async
    sys.modules["kivi_sycl._C"] = fake

    import kivi_sycl.cache.manager as manager_module
    manager_module.kivi_native = fake
    if identity:
        _install_identity_shortcut(manager_module.KiviCache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="gpt2")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--G", type=int, default=16)
    parser.add_argument("--R", type=int, default=32,
                        help="small residual on purpose: forces flushes early")
    parser.add_argument("--identity", action="store_true",
                        help="lossless flush — output must match FP32 exactly")
    parser.add_argument("--prompt", default="In a shocking finding, scientists "
                        "discovered a herd of unicorns living in a remote "
                        "valley. The unicorns were")
    args = parser.parse_args()

    install_fake_native(args.identity)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kivi_sycl import generate

    print(f"Loading {args.model} (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    print("\n--- FP32 baseline (model.generate, greedy) ---")
    with torch.no_grad():
        ref_ids = model.generate(input_ids, max_new_tokens=args.tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)[0]
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    print(ref_text)

    mode = "identity (lossless)" if args.identity else "2-bit reference"
    print(f"\n--- KIVI pipeline on CPU [{mode}, G={args.G}, R={args.R}] ---")
    kivi_text = generate(model, tokenizer, args.prompt,
                         max_new_tokens=args.tokens,
                         group_size=args.G, residual_length=args.R,
                         device="cpu", do_sample=False)
    print(kivi_text)

    kivi_ids = tokenizer(kivi_text, return_tensors="pt").input_ids[0]
    n = min(len(ref_ids), len(kivi_ids))
    match = (ref_ids[:n] == kivi_ids[:n]).float().mean().item()
    diverge = next((t for t in range(n) if ref_ids[t] != kivi_ids[t]), n)
    print(f"\nToken match vs FP32: {match * 100:.1f}%  "
          f"(first divergence at token {diverge}/{n})")

    # Compare decoded text, not re-tokenized ids: BPE re-tokenization of
    # decoded text can shift token boundaries and fake a mismatch.
    if args.identity:
        if kivi_text == ref_text:
            print("PASS: cache bookkeeping is lossless end-to-end.")
        else:
            print("FAIL: identity mode must reproduce the FP32 baseline "
                  "exactly — the cache manager dropped, reordered, or "
                  "staled tokens.")
            sys.exit(1)


if __name__ == "__main__":
    main()
