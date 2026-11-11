"""Tokenizer conformance corpus for the C++ tokenizer (Phase 5 A.5 / A.7).

Encodes a corpus with the artifact's transformers tokenizer (the reference
users run) and writes, per text, the ids with and without special tokens and
both decodings. The C++ test (tests/native/test_tokenizer.cpp) must match
every entry exactly.

    python tools/tokenizer_golden.py --artifacts artifacts/tinyllama \
        --out build-native/golden/tinyllama_tokenizer.json

Corpus: WikiText-2 test lines (offline HF cache; skipped if unavailable),
hand-written stress cases (whitespace, special tokens mid-text, Unicode
scripts, emoji, invalid-looking byte sequences, code), and this repository's
own source/markdown files cut into chunks.
"""

import argparse
import glob
import json
import os

STRESS = [
    "", " ", "  ", "   ", "\n", "\n\n", "\t", " \t\n ", "a", " a", "a ", "  a  b  ",
    "Hello world", "Hello, world!", "Once upon a time, in a small village by the sea,",
    "<s>", "</s>", "<unk>", "<s>x", "x<s>y", "a</s>b c", "<s> x", "</s></s>", "<s><s> ",
    "text <|im_start|>user\nhi<|im_end|>\n", "<|endoftext|>", "<|im_start|>",
    "naïve café résumé", "Ünïcödé ñ ß ø å", "日本語のテキスト", "中文测试，标点。",
    "한국어 문장", "Ελληνικά", "русский текст", "עברית", "العربية", "हिन्दी भाषा",
    "ไทย", "🤖🚀✨ emoji 👨‍👩‍👧‍👦 flags 🇮🇳", "zero​width", " nbsp ",
    "é combining", "ﬁ ligature", "Ⅻ roman", "½ ¾ ²", "▁ metaspace char",
    "def f(x):\n    return x ** 2  # square\n", "for (int i = 0; i < n; ++i) { a[i] = b[i]; }",
    "SELECT * FROM t WHERE x = 'y';", "https://example.com/path?q=1&r=2#frag",
    "1234567890 3.14159 -42 1e-9 0xFF", "$100.00 €50 ¥1000 £20", "I'm don't we'll they've",
    "CAPS lock MiXeD", "repeated repeated repeated", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "...!!!???---___===", "\"quoted\" 'single' `back`", "tab\tseparated\tvalues",
    "trailing spaces   ", "   leading spaces", "line1\r\nline2\rline3",
]


def corpus(max_wiki: int):
    texts = list(STRESS)
    try:
        from datasets import load_dataset
        for name in ("Salesforce/wikitext", "wikitext"):       # hub id moved
            try:
                ds = load_dataset(name, "wikitext-2-raw-v1", split="test")
                break
            except Exception:
                if name == "wikitext":
                    raise
        lines = [t for t in ds["text"] if t.strip()]
        texts += lines[:max_wiki]
    except Exception as e:                                  # offline without the cache
        print(f"warning: WikiText-2 unavailable ({type(e).__name__}); stress + repo corpus only")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(f"{root}/src/**/*.py", recursive=True)
                   + glob.glob(f"{root}/src/**/*.cpp", recursive=True)
                   + glob.glob(f"{root}/*.md"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            s = fh.read()
        texts += [s[i:i + 1500] for i in range(0, min(len(s), 6000), 1500)]
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wiki", type=int, default=2000)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.artifacts)
    cases = []
    for t in corpus(a.max_wiki):
        ids = tok(t).input_ids
        cases.append({"text": t, "ids": ids,
                      "ids_plain": tok(t, add_special_tokens=False).input_ids,
                      "decoded": tok.decode(ids),
                      "decoded_skip": tok.decode(ids, skip_special_tokens=True)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"tokenizer": a.artifacts, "cases": cases}, fh, ensure_ascii=False)
    print(f"wrote {a.out}: {len(cases)} cases, {sum(len(c['ids']) for c in cases)} tokens")


if __name__ == "__main__":
    main()
