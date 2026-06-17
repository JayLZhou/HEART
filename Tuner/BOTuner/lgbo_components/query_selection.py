"""§6 Within-cluster query selection: dynamic weighting + importance sampling (Horvitz–Thompson).

Replaces the old round-robin "sliding window" (which scored every candidate config on a
DIFFERENT non-overlapping query slice -> unfair, biased GP labels). Here:

  score   s_q = sigma_q * b_q      sigma_q = EMA/window std of f_q across tried configs
                                    b_q     = sqrt(c * log(sum_j n_j) / (n_q + 1))  (UCB bonus)
  incl.   pi_q = clip(s_q / Z, pi_min, 1),  Z solved so  sum_q pi_q = N_k
  sample  systematic pps -> subset Q_k with exact first-order inclusion prob = pi_q
  agg     F_hat_k = (1/M) sum_{q in Q_k} f_q / pi_q          (Horvitz–Thompson, UNBIASED)

Unbiasedness is decoupled from the scoring (HT corrects any inclusion design), so a noisy
sigma_q only affects variance, never bias. Cold-start (no history) or homogeneous clusters
degrade to ~uniform sampling, where HT reduces exactly to the plain mean.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np


class QuerySelector:
    def __init__(self, pi_min: float = 0.02, ucb_c: float = 2.0, hist_window: int = 12, seed: int = 42):
        self.pi_min = float(pi_min)
        self.ucb_c = float(ucb_c)
        self.hist_window = int(hist_window)
        self.rng = np.random.default_rng(seed)
        self._stats: dict = defaultdict(dict)      # cid -> {qid: {"hist": deque, "n": int}}
        self._total_n: dict = defaultdict(int)      # cid -> sum_j n_j

    # --- per-query score: config-sensitivity (sigma) x UCB exploration bonus ---
    def _score(self, cid, qid) -> float:
        st = self._stats[cid].get(qid)
        n_q = st["n"] if st else 0
        sigma = float(np.std(st["hist"])) if (st and len(st["hist"]) >= 2) else 0.0
        total = max(self._total_n[cid], 1)
        b = math.sqrt(self.ucb_c * math.log(total + 1.0) / (n_q + 1.0))
        return (sigma + 1e-9) * b                   # +eps so cold-start ranks by the UCB bonus

    # --- scores -> inclusion probs with sum == N and a feasible floor ---
    @staticmethod
    def _inclusion_probs(scores: np.ndarray, N: int, pi_min: float) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=float), 0.0, None)
        M = len(s)
        N = max(1, min(int(N), M))
        if N >= M:
            return np.ones(M)
        # floor must satisfy pi_min * M <= N, else sum can't reach N -> cap it well below uniform
        pi_min = min(float(pi_min), 0.5 * N / M)
        if not np.isfinite(s).all() or s.sum() <= 0:
            return np.full(M, N / M)                # uniform fallback
        # sum_i clip(s_i/Z, pi_min, 1) is monotone decreasing in Z -> bisection
        lo, hi = 1e-12, s.max() / max(pi_min, 1e-12) + 1.0
        for _ in range(100):
            Z = 0.5 * (lo + hi)
            tot = np.clip(s / Z, pi_min, 1.0).sum()
            if tot > N:
                lo = Z
            else:
                hi = Z
        pi = np.clip(s / (0.5 * (lo + hi)), pi_min, 1.0)
        # nudge interior entries so the sum is exactly N (systematic pps needs sum == N)
        diff = N - pi.sum()
        if abs(diff) > 1e-9:
            interior = (pi > pi_min + 1e-12) & (pi < 1.0 - 1e-12)
            if interior.any():
                pi[interior] = np.clip(pi[interior] + diff * pi[interior] / pi[interior].sum(), pi_min, 1.0)
        return pi

    # --- systematic pps: first-order inclusion prob = pi exactly (0<pi<=1, sum pi=N) ---
    def _systematic_pps(self, pi: np.ndarray, N: int) -> list:
        M = len(pi)
        order = self.rng.permutation(M)             # randomise neighbours to decorrelate
        cum = np.cumsum(pi[order])
        cum[-1] = max(float(cum[-1]), float(N))     # guard against float drift
        start = float(self.rng.random())
        sel, j = [], 0
        for pt in start + np.arange(N):
            while j < M - 1 and cum[j] < pt:
                j += 1
            sel.append(int(order[j]))
        return sorted(set(sel))

    def select(self, cid, qlist: list, N_k: int):
        """Return (selected_queries, pi_map={qid: pi_q}) for one cluster round."""
        M = len(qlist)
        if M == 0:
            return [], {}
        N_k = max(1, min(int(N_k), M))
        scores = np.array([self._score(cid, q.get("id")) for q in qlist], dtype=float)
        pi = self._inclusion_probs(scores, N_k, self.pi_min)
        idx = self._systematic_pps(pi, N_k)
        selected = [qlist[i] for i in idx]
        pi_map = {qlist[i].get("id"): float(pi[i]) for i in idx}
        return selected, pi_map

    def update(self, cid, per_query_f: dict) -> None:
        """Feed observed f_q (qid -> reward) back into per-query stats (§6.7 closed loop)."""
        cstats = self._stats[cid]
        for qid, f in per_query_f.items():
            st = cstats.get(qid)
            if st is None:
                st = {"hist": deque(maxlen=self.hist_window), "n": 0}
                cstats[qid] = st
            st["hist"].append(float(f))
            st["n"] += 1
            self._total_n[cid] += 1


def ht_estimate(per_query_f: dict, pi_map: dict, M: int, self_normalized: bool = False) -> float:
    """Horvitz–Thompson (default) / Hájek self-normalized unbiased mean over the size-M cluster.

    Under uniform pi = N/M this reduces exactly to the plain sample mean, so it is a safe drop-in.
    """
    num = den = 0.0
    for qid, f in per_query_f.items():
        pi = pi_map.get(qid)
        if not pi or pi <= 0.0:
            continue
        num += float(f) / pi
        den += 1.0 / pi
    if self_normalized:
        return num / den if den > 0 else 0.0
    return num / max(int(M), 1)
