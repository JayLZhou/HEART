import math
import torch
from typing import Any, Dict, List, Tuple, Union, Optional
import re,json,ast
from typing import Any, Dict, List, Tuple, Union, Optional

ExpertInput = Union[List[Any], Tuple[Any, ...], str]

def _normalize_expert_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\s*\[\s*([A-Za-z]+)\s*,', r'["\1",', s)
    s = re.sub(r'([\{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)
    s = s.replace("'", '"')
    return s

def _loads_forgiving(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)

PrefKind = str
def _norm_ppf(p: float) -> float:
    p = min(max(p,1e-12),1-1e-12)
    return math.sqrt(2.0)*float(torch.erfinv(torch.tensor(2.0*p-1.0)))
def confidence_to_delta(confidence: float, scale: float = 1.0, two_sided: bool = False) -> float:
    p = float(confidence)
    if two_sided:
        p = 0.5*(1.0+p)
    z = _norm_ppf(p)
    return scale*z
def _box_from_center_radius(center: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    lb = (center-radius).clamp(1e-6,1-1e-6)
    ub = (center+radius).clamp(1e-6,1-1e-6)
    return lb, ub

def decide_preference(
    *,
    kind: PrefKind,
    confidence: float,
    d: int,
    grid_size: int = 512,
    region_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    region_center: Optional[torch.Tensor] = None,
    region_radius: Optional[float] = None,
    x_star: Optional[torch.Tensor] = None,
    y_star: Optional[float] = None,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
) -> Dict[str, Any]:
    assert 0.0 < confidence < 1.0
    plan: Dict[str, Any] = {"kind_input": kind}
    delta = confidence_to_delta(confidence, scale=1.0, two_sided=False)
    plan["delta"] = float(delta)

    if kind == "region":
        assert (region_box is not None) or (region_center is not None and region_radius is not None)

        if region_box is None:
            lb, ub = _box_from_center_radius(region_center, float(region_radius))
        else:
            lb, ub = region_box
        lb = lb.clone().detach()
        ub = ub.clone().detach()
        assert lb.shape == (d,) and ub.shape == (d,)

        vol = float(torch.clamp(ub - lb, 0, 1).prod().item())
        E = grid_size * vol

        if E < E_soft_low:
            c = 0.5 * (lb + ub)
            mode = "point"
            plan.update({
                "mode": mode,
                "x_star": c, "y_star": y_star,
                "why": f"region too small for grid_size={grid_size} (E≈{E:.2f} < {E_soft_low}); fallback to point."
            })
        elif E < E_soft_high:
            mode = "region-soft"
            plan.update({
                "mode": mode,
                "region": {"lb": lb, "ub": ub, "grid_size": grid_size, "smooth": 0.08},
                "x_star": 0.5 * (lb + ub), "y_star": y_star,
                "why": f"medium region density E≈{E:.2f} in [{E_soft_low},{E_soft_high}); use soft-box."
            })
        else:
            mode = "region"
            plan.update({
                "mode": mode,
                "region": {"lb": lb, "ub": ub, "grid_size": grid_size, "smooth": None},
                "x_star": 0.5 * (lb + ub), "y_star": y_star,
                "why": f"region well-covered: E≈{E:.2f} ≥ {E_soft_high}."
            })
        return plan

    if kind == "value":
        assert y_star is not None
        plan.update({
            "mode": "value",
            "y_star": y_star,
            "why": "value prior selected; delta from confidence controls strength."
        })
        return plan

    if kind == "point":
        assert x_star is not None
        plan.update({
            "mode": "point",
            "x_star": x_star,
            "why": "point prior selected; delta from confidence controls strength."
        })
        return plan

    raise ValueError(f"unknown kind={kind}")


ExpertInput = Union[List[Any], Tuple[Any, ...]]

def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"cannot convert to float: {x!r}")

def _to_1d_tensor(items: Any, *, dtype=torch.double) -> torch.Tensor:
    if isinstance(items, torch.Tensor):
        t = items.detach().to(dtype=dtype)
    else:
        if isinstance(items, (list, tuple)):
            arr = [_to_float(v) for v in items]
        else:
            if isinstance(items, str):
                arr = [_to_float(v) for v in items.split(",")]
            else:
                raise ValueError(f"expect 1D list/tuple/str for vector, got {type(items)}")
        t = torch.tensor(arr, dtype=dtype)
    if t.ndim != 1:
        raise ValueError("expect 1D vector")
    return t.clamp(1e-6, 1 - 1e-6)

def parse_expert_input(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
) -> Dict[str, Any]:

    if not isinstance(expert, (list, tuple)) or len(expert) != 3:
        raise ValueError("expert input must be [kind, payload, confidence]")

    kind, payload, confidence = expert
    kind = str(kind).lower().strip()
    conf = _to_float(confidence)
    if not (0.0 < conf < 1.0):
        raise ValueError("confidence must be in (0,1)")

    kwargs: Dict[str, Any] = dict(
        kind=kind, confidence=conf,
        grid_size=int(grid_size),
        E_soft_low=float(E_soft_low),
        E_soft_high=float(E_soft_high),
    )

    if kind == "point":
        x_star = _to_1d_tensor(payload, dtype=dtype)
        _d = len(x_star) if d is None else d
        if d is not None and len(x_star) != d:
            raise ValueError(f"x_star length {len(x_star)} != d {d}")
        kwargs.update({"d": int(_d), "x_star": x_star})
        return kwargs

    if kind == "value":
        y_star = _to_float(payload)
        _d = int(d) if d is not None else 1
        kwargs.update({"d": _d, "y_star": float(y_star)})
        return kwargs

    if kind == "region":
        lb = ub = center = None
        radius = None

        if isinstance(payload, dict):
            if "lb" in payload and "ub" in payload:
                lb = _to_1d_tensor(payload["lb"], dtype=dtype)
                ub = _to_1d_tensor(payload["ub"], dtype=dtype)
                if len(lb) != len(ub):
                    raise ValueError("len(lb) must equal len(ub)")
            elif "center" in payload and "radius" in payload:
                center = _to_1d_tensor(payload["center"], dtype=dtype)
                radius = _to_float(payload["radius"])
            else:
                raise ValueError("region dict must have ('lb','ub') or ('center','radius')")
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            a, b = payload
            if isinstance(a, (list, tuple, str)) and isinstance(b, (list, tuple, str)):
                lb = _to_1d_tensor(a, dtype=dtype)
                ub = _to_1d_tensor(b, dtype=dtype)
                if len(lb) != len(ub):
                    raise ValueError("len(lb) must equal len(ub)")
            elif isinstance(a, (list, tuple, str)) and isinstance(b, (int, float, str)):
                center = _to_1d_tensor(a, dtype=dtype)
                radius = _to_float(b)
            else:
                raise ValueError("unrecognized region shorthand")
        else:
            raise ValueError("region payload must be dict or 2-tuple")

        if lb is not None and ub is not None:
            _d = len(lb) if d is None else d
            if d is not None and len(lb) != d:
                raise ValueError(f"lb/ub length {len(lb)} != d {d}")
            kwargs.update({"d": int(_d), "region_box": (lb, ub)})
        elif center is not None and radius is not None:
            if radius <= 0:
                raise ValueError("radius must be positive")
            _d = len(center) if d is None else d
            if d is not None and len(center) != d:
                raise ValueError(f"center length {len(center)} != d {d}")
            kwargs.update({"d": int(_d), "region_center": center, "region_radius": float(radius)})
        else:
            raise RuntimeError("internal: region parsing failed")
        return kwargs

    raise ValueError(f"unknown kind '{kind}'")

def parse_expert_input_auto(
    expert: ExpertInput,
    **kwargs_for_parse  
):

    if isinstance(expert, str):
        s = _normalize_expert_str(expert)
        expert_obj = _loads_forgiving(s)
    else:
        expert_obj = expert
    return parse_expert_input(expert_obj, **kwargs_for_parse)
def _box_width_with_clamp(center_i: float, r: float, eps: float = 1e-6) -> float:
    lb = max(center_i - r, eps)
    ub = min(center_i + r, 1.0 - eps)
    return max(ub - lb, 0.0)

def _effective_volume(center: torch.Tensor, r: float, eps: float = 1e-6) -> float:
    c = center.detach().double().cpu().tolist()
    w = 1.0
    for ci in c:
        w *= _box_width_with_clamp(float(ci), r, eps)
    return max(w, 0.0)

def choose_soft_radius_edge(
    d: int,
    grid_size: int,
    *,
    center: torch.Tensor,
    E_soft_low: float = 2.0,
    low_margin: float = 0.05,
    eps: float = 1e-6,
) -> float:

    E_target = E_soft_low + float(low_margin)
    vol_target = max(E_target / float(grid_size), 1e-12)

    lo, hi = 1e-8, 0.5 - eps

    if _effective_volume(center, hi, eps) < vol_target:
        return float(hi)

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _effective_volume(center, mid, eps) >= vol_target:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)

    E_eff = grid_size * _effective_volume(center, r, eps)
    if E_eff < E_soft_low:
        for _ in range(10):
            r = min(r * 1.2, 0.5 - eps)
            E_eff = grid_size * _effective_volume(center, r, eps)
            if E_eff >= E_soft_low:
                break
    return float(r)

def decide_preference_tilt_from_expert(
    expert: ExpertInput,
    *,
    d: Optional[int] = None,
    grid_size: int = 512,
    E_soft_low: float = 2.0,
    E_soft_high: float = 6.0,
    dtype: torch.dtype = torch.double,
    low_margin: float = 0.05,
    r_cap: float | None = None,   
) -> Dict[str, Any]:
    kw = parse_expert_input_auto(
        expert, d=d, grid_size=grid_size,
        E_soft_low=E_soft_low, E_soft_high=E_soft_high, dtype=dtype
    )
    if kw["kind"] == "point":
        x = kw["x_star"]
        dd = kw["d"]
        r = choose_soft_radius_edge(
            dd, grid_size, center=x, E_soft_low=E_soft_low, low_margin=low_margin
        )
        if r_cap is not None:
            r = min(r, float(r_cap))
        return decide_preference(
            kind="region",
            confidence=kw["confidence"],
            d=dd,
            grid_size=grid_size,
            region_center=x,
            region_radius=r,
            E_soft_low=E_soft_low,
            E_soft_high=E_soft_high,
        )
    return decide_preference(**kw)

