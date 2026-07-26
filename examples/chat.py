"""
KIVI Chat — interactive terminal chat with a 2-bit quantized KV cache
=====================================================================
An `ollama run`-style REPL: type a message, watch the reply stream back
token by token, with a live readout of how much the KV cache is compressed.

The KiviCache is kept alive across turns, so the cache grows with the whole
conversation instead of being rebuilt each time — which is the regime where
2-bit KV quantization actually pays off. Each turn only the newly rendered
tokens are fed to the model; the rest is already in the cache.

Run: python examples/chat.py [model_name] [options]

Examples:
    python examples/chat.py                          # Qwen2.5-0.5B-Instruct
    python examples/chat.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python examples/chat.py --temperature 0.9
    python examples/chat.py --no-kivi                # FP baseline, for comparison
    echo "hi" | python examples/chat.py --max-new-tokens 20   # non-interactive

Defaults to Qwen2.5-0.5B-Instruct rather than the TinyLlama used by the
benchmark scripts: measured side by side it answers "hello" with "Hello! How
can I assist you today?" where TinyLlama-1.1B-Chat produces unrelated
boilerplate, while also being 1.8x faster (9.4 vs 5.3 tok/s) and 2.2x smaller
in fp32. Compression is unaffected — at equal conversation length the two
measure 5.36x and 5.17x, since the ratio depends on tokens vs the residual
window, not on the model.

In-chat commands:
    /stats    memory + compression for the current conversation
    /reset    clear the conversation and the cache
    /help     list commands
    /quit     exit  (Ctrl-D or Ctrl-C also work)

Note this machine runs the model on CPU, so replies stream at single-digit
tokens/second. That is the model, not the cache — see
docs/status_and_roadmap.md 2.3.
"""

import argparse
import os
import sys
import time
import warnings

# Quiet the startup banner before transformers/torch are imported — the
# progress-bar setting is only read at import time. Everything suppressed here
# is informational and unconditional on this machine (the Iris Xe support
# warning fires on every run), so it buries the actual prompt. --verbose
# restores all of it.
if "--verbose" not in sys.argv:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message=".*not officially supported.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if "--verbose" not in sys.argv:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()

from kivi_sycl import KiviCache
from kivi_sycl.generation.sampling import sample_token


class ChatSession:
    """Multi-turn chat state: message list + a persistent KV cache.

    Invariant: `self.fed_ids` is exactly the token sequence the cache holds,
    in order. Every turn re-renders the whole conversation and feeds only the
    suffix beyond `fed_ids`. If the re-rendered prefix ever disagrees with
    what we fed (a chat template that rewrites earlier turns, or a
    tokenizer that merges across the boundary), we rebuild from scratch
    rather than serve a corrupted prefix — correctness over speed.
    """

    def __init__(self, model, tokenizer, use_kivi=True, G=32, R=128,
                 device="xpu", serving="rebuild", async_flush=True,
                 key_flush="drain", prefill_chunk=None):
        self.model = model
        self.tokenizer = tokenizer
        self.use_kivi = use_kivi
        self._cache_kwargs = dict(device=device, group_size=G,
                                  residual_length=R, serving=serving,
                                  async_flush=async_flush,
                                  key_flush=key_flush,
                                  prefill_chunk=prefill_chunk)
        self.messages = []
        self._new_cache()

    def _new_cache(self):
        self.cache = (KiviCache.from_model(self.model, **self._cache_kwargs)
                      if self.use_kivi else None)
        self.fp_past = None          # only used when use_kivi is False
        self.fed_ids = None
        self.n_fed = 0

    def reset(self, keep_messages=False):
        if not keep_messages:
            self.messages = []
        if self.cache is not None:
            self.cache.reset()
        self._new_cache()

    # -- prompt rendering ------------------------------------------------- #

    def _render_ids(self):
        """Token ids for the whole conversation, ending in the cue for the
        assistant's next turn."""
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True)
        else:
            # Base model with no template — plain transcript. Quality will be
            # worse than an instruction-tuned checkpoint; that is expected.
            parts = [f"{m['role'].capitalize()}: {m['content']}"
                     for m in self.messages]
            text = "\n".join(parts) + "\nAssistant:"
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).input_ids
        # Chat templates render the role markers but not BOS, and
        # add_special_tokens=True cannot be used here — it would insert a BOS
        # into every per-turn delta, mid-conversation. Prepend it once, which
        # keeps the sequence prefix-stable for the incremental-feed guard.
        # Llama-family models are trained with BOS present and answer measurably
        # better with it ("Yes, 1+2 is equal to 3." -> "Yes, 1 + 2 is 3.").
        bos = self.tokenizer.bos_token_id
        if bos is not None and (ids.shape[1] == 0 or ids[0, 0].item() != bos):
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        return ids

    # -- model plumbing --------------------------------------------------- #

    def _feed(self, ids):
        """Forward `ids`, append them to the cache, return last-position logits."""
        with torch.no_grad():
            if self.use_kivi:
                # An empty cache has no tensors to serve; pass None on the
                # very first forward rather than a zero-length past.
                past = self.cache.get_full_cache() if self.n_fed else None
                out = self.model(ids, past_key_values=past, use_cache=True)
                self.cache.add_tokens(out.past_key_values)
            else:
                out = self.model(ids, past_key_values=self.fp_past, use_cache=True)
                self.fp_past = out.past_key_values

        self.fed_ids = ids if self.fed_ids is None else \
            torch.cat([self.fed_ids, ids], dim=1)
        self.n_fed += ids.shape[1]
        return out.logits[:, -1, :]

    def send(self, user_text, max_new_tokens=256, temperature=0.7,
             top_k=50, do_sample=True, stream=True):
        """Append a user turn, stream the reply, return (reply, tok_per_s)."""
        self.messages.append({"role": "user", "content": user_text})

        ids = self._render_ids()
        if self.n_fed and not torch.equal(ids[:, :self.n_fed], self.fed_ids):
            # See the class docstring — rebuild instead of trusting the prefix.
            self.reset(keep_messages=True)
            ids = self._render_ids()
        new_ids = ids[:, self.n_fed:]
        if new_ids.shape[1] == 0:
            new_ids = ids[:, -1:]        # degenerate; keep the forward well-formed

        logits = self._feed(new_ids)

        eos_id = self.tokenizer.eos_token_id
        gen_ids, printed = [], ""
        start = time.time()
        for _ in range(max_new_tokens):
            next_id = sample_token(logits, temperature, top_k, do_sample)
            if eos_id is not None and next_id.item() == eos_id:
                break
            gen_ids.append(next_id.item())

            if stream:
                # Decode the whole reply each step and print only the delta:
                # a single BPE piece is often not valid text on its own, so
                # decoding token-by-token would emit mojibake mid-word.
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                sys.stdout.write(text[len(printed):])
                sys.stdout.flush()
                printed = text

            logits = self._feed(next_id)

        elapsed = time.time() - start
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if stream:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.messages.append({"role": "assistant", "content": reply})
        rate = len(gen_ids) / elapsed if elapsed > 0 and gen_ids else 0.0
        return reply, rate

    # -- reporting -------------------------------------------------------- #

    def stats_line(self):
        if not self.use_kivi:
            return f"FP baseline (no KIVI) — {self.n_fed} tokens cached"
        if self.n_fed == 0:
            return "cache empty"
        self.cache.harvest_pending()
        s = self.cache.get_stats(layer=0)
        layers = self.cache.num_layers
        kivi_mb = (s["kivi_bytes"] + s["serving_copy_bytes"]) * layers / 1024 / 1024
        fp_mb = s["fp_baseline_bytes"] * layers / 1024 / 1024
        line = (f"{s['total_tokens']} tokens "
                f"({s['quantized_tokens']} quantized + {s['residual_tokens']} FP) | "
                f"KV cache {kivi_mb:.1f} MB vs {fp_mb:.1f} MB FP "
                f"= {s['compression']:.2f}x compression")
        if s["quantized_tokens"] == 0:
            # Expected, not a failure: nothing is quantized until the
            # conversation outgrows the residual window.
            line += (f" (nothing quantized yet — the whole conversation still "
                     f"fits in the R={self.cache.R} FP residual)")
            return line

        # Second line only once there is something to report.
        #
        # `mean_compression` is labelled "session avg" here, NOT
        # "time-averaged" as in the benchmarks, and the difference matters. In
        # eval_longctx the context is fixed, so the average removes the
        # residual window's sawtooth and is the figure to compare configs
        # with. A chat conversation instead grows from empty, so this average
        # is taken across every context length the session passed through —
        # including the early turns that fit entirely in the FP residual and
        # compressed ~1x. It will therefore sit *below* the instantaneous
        # figure and drift up as the conversation lengthens. That is correct,
        # but it is a session history, not a better estimate of the current
        # state. Do not quote it as a headline compression number.
        peak_mb = s["peak_bytes"] * layers / 1024 / 1024
        line += (f"\n  {s['mean_compression']:.2f}x session avg "
                 f"(incl. early short turns) | "
                 f"peak {peak_mb:.1f} MB ({s['peak_compression']:.2f}x vs FP) | "
                 f"key window {s['mean_residual_tokens']:.0f} tok avg "
                 f"[{s['key_flush']}]")
        return line


BANNER = """\
KIVI Chat — 2-bit quantized KV cache. /help for commands, /quit to exit.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace chat model (default: Qwen2.5-0.5B-Instruct)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    p.add_argument("--G", type=int, default=32, help="quantization group size")
    p.add_argument("--R", type=int, default=128, help="FP residual window")
    p.add_argument("--serving", choices=["rebuild", "cached"], default="rebuild")
    p.add_argument("--key-flush", choices=["drain", "reset"], default="drain",
                   dest="key_flush",
                   help="reset is the paper's key window: quantize the whole "
                        "key buffer at R and leave it empty, averaging R/2 "
                        "instead of ~R. Compresses better. Requires R %% G == 0")
    p.add_argument("--prefill-chunk", type=int, default=None,
                   dest="prefill_chunk",
                   help="ingest long prompts in slices of this many tokens to "
                        "bound peak memory (packed output is unchanged). Must "
                        "be a multiple of G and >= R")
    p.add_argument("--no-kivi", action="store_true",
                   help="run a plain FP cache, for side-by-side comparison")
    p.add_argument("--system", type=str, default=None,
                   help="system prompt. Small chat models follow instructions "
                        "much better with one — try \"Answer directly and "
                        "concisely.\"")
    p.add_argument("--quiet", action="store_true", help="hide the per-turn stats line")
    p.add_argument("--verbose", action="store_true",
                   help="show torch/transformers startup warnings and progress bars")
    args = p.parse_args()

    use_kivi = not args.no_kivi
    if use_kivi and not torch.xpu.is_available():
        print("Error: XPU not found (needed for the quantization kernels). "
              "Use --no-kivi to run the FP baseline.")
        sys.exit(1)

    print(f"Loading {args.model} ...", flush=True)
    # Try the local cache first. Besides being faster and working offline, this
    # skips the Hub round-trip that prints an unauthenticated-request banner to
    # stderr on every start (emitted by the native hf_xet client, so it cannot
    # be silenced with warnings.filterwarnings).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    session = ChatSession(model, tokenizer, use_kivi=use_kivi, G=args.G, R=args.R,
                          serving=args.serving, key_flush=args.key_flush,
                          prefill_chunk=args.prefill_chunk)
    if args.system:
        session.messages.append({"role": "system", "content": args.system})

    if use_kivi:
        opts = [f"G={args.G}", f"R={args.R}", f"serving={args.serving}",
                f"key_flush={args.key_flush}"]
        if args.prefill_chunk:
            opts.append(f"prefill_chunk={args.prefill_chunk}")
        mode = f"KIVI 2-bit ({', '.join(opts)})"
    else:
        mode = "FP baseline"
    print(f"{BANNER}Mode: {mode}\n")

    interactive = sys.stdin.isatty()
    while True:
        try:
            line = input("you> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            if not interactive:
                break
            continue

        cmd = line.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/help":
            print("  /stats  memory + compression\n"
                  "  /reset  clear conversation and cache\n"
                  "  /quit   exit")
            continue
        if cmd == "/stats":
            print(f"  {session.stats_line()}")
            continue
        if cmd == "/reset":
            session.reset()
            if args.system:
                session.messages.append({"role": "system", "content": args.system})
            print("  conversation and cache cleared")
            continue

        print("bot> ", end="", flush=True)
        try:
            _, rate = session.send(line, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k,
                                   do_sample=not args.greedy)
        except KeyboardInterrupt:
            # Mid-generation Ctrl-C: the cache and the message list are now out
            # of step (a partial reply was fed but never recorded), so drop the
            # turn rather than leave the session inconsistent.
            print("\n  [interrupted — resetting cache to stay consistent]")
            session.messages.pop()
            session.reset(keep_messages=True)
            continue

        if not args.quiet:
            print(f"  [{rate:.1f} tok/s | {session.stats_line()}]")


if __name__ == "__main__":
    main()
