#!/usr/bin/env python3
"""Build the fixed 30-document TriviaQA test set used by the F1-accuracy experiment.

Run this ONCE on the GPU/server side. It downloads TriviaQA from HuggingFace,
selects a deterministic set of long documents (seed 1234, so every run picks the
same 30 documents), assigns each a stable ``mzcache_id`` (1..30, in alphabetical
order of ``doc_id``), and writes a single JSON file that all three tools consume:

    mz_accuracy            (GPU)   prefill + save .kv per doc, score original F1
    mz_save_state_v2       (phone) compress/decompress each .kv on-device
    mz_load_state_accuracy (GPU)   score F1 on the compressed .kv

Usage:
    pip install datasets
    python3 prepare_triviaqa.py [config] [out.json]
      config   TriviaQA config (default "rc")
      out.json output path (default triviaqa_30docs_seed1234.json)

The mzcache_id is what ties a document to its .kv file across the two machines,
so DO NOT reshuffle it — the phone names states by mzcache_id and the GPU scorer
matches them back by the same id.
"""
import json
import sys
import random

# ---- selection parameters (do not change: they define the fixed test set) ----
MIN_CONTEXT_CHARS = 40000
MAX_CONTEXT_CHARS = 140000
MAX_DOCS          = 30
SPLIT             = "train"
RANDOM_SEED       = 1234
# ------------------------------------------------------------------------------

CONFIG   = sys.argv[1] if len(sys.argv) > 1 else "rc"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "triviaqa_30docs_seed1234.json"


def main():
    from datasets import load_dataset

    random.seed(RANDOM_SEED)
    data = load_dataset("mandarjoshi/trivia_qa", CONFIG)[SPLIT]

    # Collect every document whose wiki context is within the length band, keyed
    # by doc_id, gathering all question/answer pairs that reference it.
    docs = {}
    for ex in data:
        q_text = (ex.get("question") or "").strip()
        q_id   = ex.get("question_id")

        ans = ex.get("answer") or {}
        answers = []
        v = (ans.get("value") or "").strip()
        if v:
            answers.append(v)
        for al in ans.get("aliases") or []:
            s = (al or "").strip()
            if s:
                answers.append(s)
        answers = list(dict.fromkeys(answers))  # dedupe, keep order

        if not q_text or not answers:
            continue

        ep        = ex.get("entity_pages") or {}
        titles    = ep.get("title") or []
        filenames = ep.get("filename") or []
        contexts  = ep.get("wiki_context") or []

        for title, fname, ctx in zip(titles, filenames, contexts):
            ctx = (ctx or "").strip()
            if not ctx or len(ctx) < MIN_CONTEXT_CHARS or len(ctx) > MAX_CONTEXT_CHARS:
                continue
            doc_id = fname or title or f"wiki::{hash(ctx)}"
            docs.setdefault(doc_id, {"doc_id": doc_id, "context": ctx, "qas": []})
            docs[doc_id]["qas"].append({"q_id": q_id, "input": q_text, "answers": answers})

    # Deterministically sample MAX_DOCS of them.
    selected_ids = random.sample(list(docs.keys()), min(MAX_DOCS, len(docs)))
    selected = [docs[i] for i in selected_ids]

    # Assign mzcache_id 1..N in alphabetical doc_id order (stable across runs).
    selected.sort(key=lambda d: d["doc_id"])
    for idx, item in enumerate(selected, start=1):
        item["mzcache_id"] = idx
        print(f"  mzcache_id {idx:2d}  <-  {item['doc_id']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(selected)} documents (of {len(docs)} valid) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
