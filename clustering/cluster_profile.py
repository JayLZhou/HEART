#!/usr/bin/env python3
"""Cluster on ONLY the 12 config-aligned LLM profile dimensions (no embedding, no NER).

Output: datasets/<OUT_DS>/Question.json (new cluster_id) + Corpus.json copy.
"""
import json
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SRC = os.environ.get("SRC_DS", "datasets/hotpotqa_1000_c5_real")
OUT_NAME = os.environ.get("OUT_DS", "hotpotqa_1000_c5_profile")
PROF = os.environ.get("PROF_FILE", "clustering/data/profile_llm.json")
K = int(os.environ.get("K", "5"))
SEED = int(os.environ.get("SEED", "42"))
OUT_DIR = os.path.join("datasets", OUT_NAME)

DIMS = ["n_facts", "n_hops", "compositional", "comparison", "exact_term_need",
        "entity_density", "numeric_temporal", "paraphrase_gap", "reasoning_depth",
        "ambiguity", "distractor_risk", "answer_length"]


def main():
    rows = [json.loads(l) for l in open(os.path.join(SRC, "Question.json")) if l.strip()]
    questions = [str(r["question"]) for r in rows]
    prof = {p["qid"]: p for p in json.load(open(PROF))}
    raw_all = np.array([[float(prof[r["qid"]].get(c, 0)) for c in DIMS] for r in rows], dtype=np.float32)

    # drop near-constant dims (a single value covers >95%): they carry no signal and,
    # once z-scored, let the rare minority dominate distances as outliers.
    keep = []
    for j, name in enumerate(DIMS):
        col = raw_all[:, j]
        vals, cnts = np.unique(col, return_counts=True)
        mode_frac = cnts.max() / len(col)
        (keep.append(j) if mode_frac <= 0.95 else
         print(f"[main] drop near-constant dim '{name}' (mode covers {mode_frac:.0%})", flush=True))
    dims = [DIMS[j] for j in keep]
    raw = raw_all[:, keep]
    X = StandardScaler().fit_transform(raw)
    print(f"[main] {len(rows)} queries, clustering on {len(dims)}/{len(DIMS)} profile dims: {dims}, KMeans k={K}", flush=True)

    labels = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X)
    sizes = np.bincount(labels, minlength=K)
    print(f"[main] cluster sizes = {sizes.tolist()}  (min={sizes.min()} max={sizes.max()})", flush=True)

    gmean, gstd = raw.mean(0), raw.std(0) + 1e-8
    print("\n[profile] per-cluster mean of each dim (z-dev in parens):")
    for c in range(K):
        idx = np.where(labels == c)[0]
        cm = raw[idx].mean(0)
        z = (cm - gmean) / gstd
        parts = ", ".join(f"{dims[j]}={cm[j]:.1f}({'+' if z[j] >= 0 else ''}{z[j]:.1f})"
                          for j in sorted(range(len(dims)), key=lambda j: -abs(z[j]))[:5])
        print(f"  c{c} (n={len(idx)}): {parts}")
        print(f"        e.g. {questions[idx[0]][:85]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "Question.json"), "w") as f:
        for r, lab in zip(rows, labels):
            o = dict(r); o["cluster_id"] = int(lab)
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    shutil.copy(os.path.join(SRC, "Corpus.json"), os.path.join(OUT_DIR, "Corpus.json"))
    print(f"\n[main] wrote {OUT_DIR}/Question.json (+ Corpus.json)", flush=True)


if __name__ == "__main__":
    main()
