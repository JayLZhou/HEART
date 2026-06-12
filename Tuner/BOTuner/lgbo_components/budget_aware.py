from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


@dataclass
class ClusterUtilityState:
    pot_ema: float = 0.0
    syn_ema: float = 0.0


@dataclass
class ClusterRoundStats:
    cluster_id: str
    confidence: float
    region_center: List[float]
    region_half_width: List[float]
    region_points: List[List[float]]
    posterior_var_mean: float
    region_effective_size: float
    pot_raw: float = 0.0
    syn_raw: float = 0.0
    pot_norm: float = 0.0
    syn_norm: float = 0.0
    utility: float = 0.0


@dataclass
class BudgetAwareAllocator:
    n_min: int = 1
    tau: float = 1.0
    ema_alpha: float = 0.9
    warm_start_synergy_threshold: float = 0.6
    rkhs_lengthscale: float = 0.2
    _ema_state: Dict[str, ClusterUtilityState] = field(default_factory=dict)

    def _ensure_state(self, cluster_ids: Iterable[str]) -> None:
        for cid in cluster_ids:
            if cid not in self._ema_state:
                self._ema_state[cid] = ClusterUtilityState()

    @staticmethod
    def _normalize(values: Dict[str, float], lo: float = 0.0, hi: float = 1.0) -> Dict[str, float]:
        if not values:
            return {}
        vals = list(values.values())
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-12:
            mid = (lo + hi) / 2.0
            return {k: mid for k in values}
        scale = (hi - lo) / (vmax - vmin)
        return {k: lo + (v - vmin) * scale for k, v in values.items()}

    def _rbf_kernel(self, Xa: torch.Tensor, Xb: torch.Tensor) -> torch.Tensor:
        ell = max(float(self.rkhs_lengthscale), 1e-6)
        d2 = torch.cdist(Xa, Xb, p=2.0) ** 2
        return torch.exp(-0.5 * d2 / (ell * ell))

    def _rkhs_cosine(self, pa: Sequence[Sequence[float]], pb: Sequence[Sequence[float]]) -> float:
        if not pa or not pb:
            return 0.0
        Xa = torch.tensor(pa, dtype=torch.double)
        Xb = torch.tensor(pb, dtype=torch.double)
        if Xa.ndim != 2 or Xb.ndim != 2 or Xa.shape[1] != Xb.shape[1]:
            return 0.0
        na = Xa.shape[0]
        nb = Xb.shape[0]
        aa = torch.full((na, 1), 1.0 / na, dtype=torch.double)
        bb = torch.full((nb, 1), 1.0 / nb, dtype=torch.double)
        Kaa = self._rbf_kernel(Xa, Xa)
        Kbb = self._rbf_kernel(Xb, Xb)
        Kab = self._rbf_kernel(Xa, Xb)
        num = (aa.t() @ Kab @ bb).squeeze().item()
        den_a = (aa.t() @ Kaa @ aa).squeeze().item()
        den_b = (bb.t() @ Kbb @ bb).squeeze().item()
        den = math.sqrt(max(den_a, 1e-12) * max(den_b, 1e-12))
        if den <= 1e-12:
            return 0.0
        return float(num / den)

    def estimate_utilities(self, cluster_stats: Dict[str, ClusterRoundStats]) -> Dict[str, ClusterRoundStats]:
        cluster_ids = list(cluster_stats.keys())
        self._ensure_state(cluster_ids)

        for cid, stat in cluster_stats.items():
            stat.pot_raw = float(max(stat.posterior_var_mean, 0.0) * max(stat.confidence, 0.0) * max(stat.region_effective_size, 0.0))

        for cid, stat in cluster_stats.items():
            others = [j for j in cluster_ids if j != cid]
            if not others:
                stat.syn_raw = 0.0
                continue
            syn = [self._rkhs_cosine(stat.region_points, cluster_stats[j].region_points) for j in others]
            stat.syn_raw = float(sum(syn) / len(syn)) if syn else 0.0

        pot_smoothed: Dict[str, float] = {}
        syn_smoothed: Dict[str, float] = {}
        for cid, stat in cluster_stats.items():
            state = self._ema_state[cid]
            state.pot_ema = self.ema_alpha * state.pot_ema + (1.0 - self.ema_alpha) * stat.pot_raw
            state.syn_ema = self.ema_alpha * state.syn_ema + (1.0 - self.ema_alpha) * stat.syn_raw
            pot_smoothed[cid] = state.pot_ema
            syn_smoothed[cid] = state.syn_ema

        pot_norm = self._normalize(pot_smoothed, 0.0, 1.0)
        syn_norm = self._normalize(syn_smoothed, -1.0, 1.0)

        for cid, stat in cluster_stats.items():
            stat.pot_norm = pot_norm[cid]
            stat.syn_norm = syn_norm[cid]
            stat.utility = stat.pot_norm + stat.syn_norm
        return cluster_stats

    def allocate(self, *, B: int, cluster_ids: Sequence[str], utilities: Dict[str, float], cold_start: bool = False) -> Dict[str, int]:
        K = len(cluster_ids)
        if K == 0:
            return {}
        if B < K * self.n_min:
            floor = max(0, B // K)
            alloc = {cid: floor for cid in cluster_ids}
            rem = B - floor * K
            for cid in cluster_ids[:rem]:
                alloc[cid] += 1
            return alloc

        if cold_start:
            base = B // K
            alloc = {cid: base for cid in cluster_ids}
            rem = B - base * K
            for cid in cluster_ids[:rem]:
                alloc[cid] += 1
            return alloc

        tau = max(self.tau, 1e-6)
        logits = torch.tensor([utilities[cid] / tau for cid in cluster_ids], dtype=torch.double)
        probs = torch.softmax(logits, dim=0).tolist()

        residual = B - K * self.n_min
        raw = [self.n_min + residual * p for p in probs]
        ints = [math.floor(v) for v in raw]
        frac = [v - math.floor(v) for v in raw]

        for i in range(len(ints)):
            if random.random() < frac[i]:
                ints[i] += 1

        delta = B - sum(ints)
        order = sorted(range(len(cluster_ids)), key=lambda i: frac[i], reverse=(delta > 0))
        idx = 0
        while delta != 0 and order:
            i = order[idx % len(order)]
            if delta > 0:
                ints[i] += 1
                delta -= 1
            else:
                if ints[i] > self.n_min:
                    ints[i] -= 1
                    delta += 1
            idx += 1
            if idx > 10_000:
                break

        return {cid: int(v) for cid, v in zip(cluster_ids, ints)}

    def best_transfer_source(
        self,
        *,
        target_cluster_id: str,
        cluster_stats: Dict[str, ClusterRoundStats],
        cluster_best: Dict[str, Dict[str, Any] | None],
    ) -> Tuple[str | None, float]:
        target = cluster_stats.get(target_cluster_id)
        if target is None:
            return None, 0.0
        best_cid: str | None = None
        best_syn = -1.0
        for cid, stat in cluster_stats.items():
            if cid == target_cluster_id:
                continue
            if not cluster_best.get(cid):
                continue
            syn = self._rkhs_cosine(target.region_points, stat.region_points)
            if syn > best_syn:
                best_syn = syn
                best_cid = cid
        if best_cid is None or best_syn < self.warm_start_synergy_threshold:
            return None, best_syn
        return best_cid, best_syn
