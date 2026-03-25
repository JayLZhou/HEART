
from __future__ import annotations
from typing import Sequence, List, Tuple, Dict, Union, Optional
import math
import random

from dataclasses import dataclass
Vec = Sequence[float]
Batch = Sequence[Vec]
Result = Dict[str, float]
BatchResult = List[Result]


DEFAULT_BOUNDS = {
    "rastrigin": (-5.12, 5.12),
    "ackley": (-5.0, 5.0),
    "griewank": (-600.0, 600.0),
    "levy": (-10.0, 10.0),
}

def _is_batch(x: Union[Vec, Batch]) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))

def _clip_to_bounds(x: Vec, bounds: Tuple[float, float]) -> List[float]:
    lo, hi = bounds
    return [min(max(float(v), lo), hi) for v in x]


def rastrigin(x: Vec) -> float:
    d = len(x)
    A = 10.0
    return A * d + sum((xi * xi - A * math.cos(2.0 * math.pi * xi)) for xi in x)

def ackley(x: Vec) -> float:

    d = len(x)
    if d == 0:
        return 0.0
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2.0 * math.pi * xi) for xi in x)
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(s1 / d))
    term2 = -math.exp(s2 / d)
    return term1 + term2 + 20.0 + math.e

def griewank(x: Vec) -> float:

    d = len(x)
    sum_term = sum((xi * xi) for xi in x) / 4000.0
    prod_term = 1.0
    for i, xi in enumerate(x, start=1):
        prod_term *= math.cos(xi / math.sqrt(float(i)))
    return 1.0 + sum_term - prod_term

def levy(x: Vec) -> float:

    d = len(x)
    if d == 0:
        return 0.0
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
    term0 = math.sin(math.pi * w[0]) ** 2
    mid = 0.0
    for i in range(d - 1):
        wi = w[i]
        mid += (wi - 1.0) ** 2 * (1.0 + 10.0 * (math.sin(math.pi * wi + 1.0) ** 2))
    termd = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return term0 + mid + termd


def toy_results(
    func_name: str,
    x: Union[Vec, Batch],
    *,
    noise_std: float = 0.0,
    seed: int | None = None,
    clip_to_domain: bool = True,
    bounds: Tuple[float, float] | None = None,
) -> Union[Result, BatchResult]:

    name = func_name.strip().lower()
    if name not in {"rastrigin", "ackley", "griewank", "levy"}:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")

    if seed is not None:
        random.seed(seed)


    f_map = {
        "rastrigin": rastrigin,
        "ackley": ackley,
        "griewank": griewank,
        "levy": levy,
    }
    f_eval = f_map[name]
    dom_bounds = bounds if bounds is not None else DEFAULT_BOUNDS[name]

    def _eval_one(xi: Vec, idx: int) -> Result:
        x_eff = list(xi)
        if clip_to_domain:
            x_eff = _clip_to_bounds(x_eff, dom_bounds)
        v = float(f_eval(x_eff))
        if noise_std > 0.0:

            if seed is not None:
                random.seed(seed * 1000 + idx)
            v += random.gauss(0.0, noise_std)
        return {"f": v}

    if _is_batch(x):
        out: BatchResult = []
        for i, xi in enumerate(x):
            out.append(_eval_one(xi, i))
        return out
    else:
        return _eval_one(x, 0)



def get_bo_bounds(func_name: str, d: int) -> List[Tuple[float, float]]:
    name = func_name.strip().lower()
    if name == "rastrigin":
        lo, hi = -5.12, 5.12
    elif name == "ackley":
        lo, hi = -5.0, 5.0
    elif name == "griewank":
        lo, hi = -600.0, 600.0
    elif name == "levy":
        lo, hi = -10.0, 10.0
    else:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")
    return [(lo, hi)] * d

@dataclass
class BONormalizer:

    bounds: List[Tuple[float, float]]  # [(lo,hi)] * d

    def __post_init__(self):
        self.bounds = [(float(lo), float(hi)) for lo, hi in self.bounds]
        if any(lo >= hi for lo, hi in self.bounds):
            raise ValueError("Invalid bounds: require lo < hi")
        self.d = len(self.bounds)


    def normalize_point(self, x: Sequence[float]) -> List[float]:
        if len(x) != self.d:
            raise ValueError(f"Dim mismatch: got {len(x)} but d={self.d}")
        out = []
        for (lo, hi), xi in zip(self.bounds, x):
            xi = float(xi)

            xi = min(max(xi, lo), hi)
            out.append((xi - lo) / (hi - lo))
        return out

    def denormalize_point(self, z: Sequence[float]) -> List[float]:
        if len(z) != self.d:
            raise ValueError(f"Dim mismatch: got {len(z)} but d={self.d}")
        out = []
        for (lo, hi), zi in zip(self.bounds, z):
            zi = float(zi)

            zi = min(max(zi, 0.0), 1.0)
            out.append(lo + zi * (hi - lo))
        return out


    def normalize_region(self, lb: Sequence[float], ub: Sequence[float]) -> Tuple[List[float], List[float]]:
        if len(lb) != self.d or len(ub) != self.d:
            raise ValueError("Dim mismatch in region")
        lb_n = self.normalize_point(lb)
        ub_n = self.normalize_point(ub)

        lb_n = [min(a, b) for a, b in zip(lb_n, ub_n)]
        ub_n = [max(a, b) for a, b in zip(lb_n, ub_n)]
        return lb_n, ub_n

    def denormalize_region(self, lb_n: Sequence[float], ub_n: Sequence[float]) -> Tuple[List[float], List[float]]:
        if len(lb_n) != self.d or len(ub_n) != self.d:
            raise ValueError("Dim mismatch in region")

        lb_n = [min(max(float(v), 0.0), 1.0) for v in lb_n]
        ub_n = [min(max(float(v), 0.0), 1.0) for v in ub_n]

        lb_n = [min(a, b) for a, b in zip(lb_n, ub_n)]
        ub_n = [max(a, b) for a, b in zip(lb_n, ub_n)]
        lb = self.denormalize_point(lb_n)
        ub = self.denormalize_point(ub_n)
        return lb, ub


def make_bo_normalizer(func_name: str, d: int, bounds_override: Optional[List[Tuple[float, float]]] = None) -> BONormalizer:
    return BONormalizer(bounds_override if bounds_override is not None else get_bo_bounds(func_name, d))


if __name__ == "__main__":

    norm = make_bo_normalizer("rastrigin", d=3)


    x_real = [0.0, -5.12, 5.12]
    z = norm.normalize_point(x_real)
    x_back = norm.denormalize_point(z)
    print("=== Point Test ===")
    print("real:", x_real)
    print("normalized:", z)
    print("denormalized:", x_back)
    print("norm.bounds:", norm.bounds)


    lb_real = [-5.12, -2.0, 0.0]
    ub_real = [5.12,  3.0, 2.0]
    lb_n, ub_n = norm.normalize_region(lb_real, ub_real)
    lb_back, ub_back = norm.denormalize_region(lb_n, ub_n)
    print("\n=== Region Test ===")
    print("real region:", (lb_real, ub_real))
    print("normalized region:", (lb_n, ub_n))