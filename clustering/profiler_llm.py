#!/usr/bin/env python3
"""Config-aligned LLM query profiler (graphrag env, LLM via 8001).

One LLM call per query -> 12 profile dimensions, each aligned to a HEART search-space knob.
Deployable (query-side only, no answer needed). Output: profile_llm.json keyed by qid.
Clustering then uses ONLY these 12 dimensions.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

QPATH = sys.argv[1] if len(sys.argv) > 1 else "datasets/hotpotqa_1000_c5_real/Question.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "clustering/data/profile_llm.json"
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8001/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-7b")
WORKERS = int(os.environ.get("WORKERS", "16"))

DB_META = ("Knowledge base = English Wikipedia passages. Questions are multi-hop factoid "
           "questions (HotpotQA) often needing facts combined from two or more passages.")

# (key, low, high, default) — every dim maps to a HEART config knob
DIMS = [
    ("n_facts", 1, 10, 1),            # -> top_k
    ("n_hops", 1, 5, 1),              # -> num_queries
    ("compositional", 0, 1, 0),       # -> query_decomposition_enabled
    ("comparison", 0, 1, 0),          # -> decomposition
    ("exact_term_need", 0, 2, 1),     # -> method / bm25_weight
    ("entity_density", 0, 2, 1),      # -> bm25_weight
    ("numeric_temporal", 0, 1, 0),    # -> sparse/BM25
    ("paraphrase_gap", 0, 2, 1),      # -> dense vs sparse
    ("reasoning_depth", 0, 1, 0),     # -> reranker
    ("ambiguity", 0, 1, 0),           # -> reranker
    ("distractor_risk", 0, 1, 0),     # -> fusion_mode
    ("answer_length", 1, 5, 1),       # -> reranker_top_k / synthesis
]

SYS = "You are a query analyzer for a retrieval-augmented generation system. Respond with ONLY one JSON object, no prose."

PROMPT = """Analyze the RAG query and rate 12 dimensions. {meta}

Query: "{q}"

Return ONE JSON object with EXACTLY these integer keys (use the stated ranges):
- "n_facts" (1-10): distinct standalone facts needed to fully answer
- "n_hops" (1-5): sequential reasoning steps / chained lookups required
- "compositional" (0 or 1): 1 if it must be broken into sub-questions
- "comparison" (0 or 1): 1 if it compares/contrasts two or more entities
- "exact_term_need" (0-2): reliance on exact rare terms; 0=none,1=some,2=high
- "entity_density" (0-2): how entity-heavy the query is; 0=low,1=med,2=high
- "numeric_temporal" (0 or 1): 1 if the answer is or hinges on a number/date
- "paraphrase_gap" (0-2): expected wording mismatch between query and source; 0=low,1=med,2=high
- "reasoning_depth" (0 or 1): 0=shallow lookup, 1=deep reasoning
- "ambiguity" (0 or 1): 1 if the query is underspecified/ambiguous
- "distractor_risk" (0 or 1): 1 if many similar/near-duplicate passages likely exist
- "answer_length" (1-5): expected answer length; 1=short span ... 5=long explanation

JSON:"""

client = OpenAI(api_key="sk-local", base_url=LLM_URL)


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"no json: {txt[:120]}")
    d = json.loads(m.group(0))
    out = {}
    for k, lo, hi, dv in DIMS:
        try:
            v = int(round(float(d.get(k, dv))))
        except Exception:
            v = dv
        out[k] = max(lo, min(hi, v))
    return out


def profile_one(row):
    q = str(row["question"])
    for attempt in range(3):
        try:
            rsp = client.chat.completions.create(
                model=LLM_MODEL, temperature=0.0, max_tokens=200,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": PROMPT.format(meta=DB_META, q=q)}],
            )
            p = parse(rsp.choices[0].message.content)
            p["qid"] = row["qid"]
            return p
        except Exception as e:
            if attempt == 2:
                p = {k: dv for k, _, _, dv in DIMS}
                p["qid"] = row["qid"]; p["_err"] = str(e)[:80]
                return p


def main():
    rows = [json.loads(l) for l in open(QPATH) if l.strip()]
    print(f"[prof] {len(rows)} queries, {WORKERS} workers, {len(DIMS)} dims -> {LLM_MODEL}", flush=True)
    out = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, p in zip(range(len(rows)), ex.map(profile_one, rows)):
            out[i] = p; done += 1
            if done % 100 == 0:
                print(f"[prof] {done}/{len(rows)}", flush=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    errs = sum(1 for p in out if p.get("_err"))
    import statistics as st
    print(f"[prof] wrote {len(out)} -> {OUT}  (errors={errs})", flush=True)
    for k, _, _, _ in DIMS:
        print(f"   {k:18s} mean={st.mean(p[k] for p in out):.2f}", flush=True)


if __name__ == "__main__":
    main()
