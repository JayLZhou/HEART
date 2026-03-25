from prior_monte_carlo import (
    ValuePeakPrior,
    PointBumpPrior,
    WeightedQLogEI,
    WeightedQEI,
    WeightedTS
)
import torch
from torch import Tensor
from typing import Type, Optional
from prior import (
    _make_sobol_grid_norm,
    _box_mask_norm,
    LinearExponentialRegionalMeanTiltPlugAndPlay,
    TiltedModel,
)
from decide import decide_preference
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
import math

@torch.no_grad()
def _greedy_select_with_min_dist(acq_fn, C: Tensor, n: int, r_min: Optional[float], *, chunk_size: int = 256) -> Tensor:
    M = C.size(0)
    device = C.device
    mask = torch.ones(M, dtype=torch.bool, device=device)
    selected_idx = []
    for _ in range(n):
        active = torch.arange(M, device=device)[mask]
        if active.numel() == 0:
            break
        best_val = None
        best_j_rel = None
        for start in range(0, active.numel(), chunk_size):
            end = min(start + chunk_size, active.numel())
            idx_chunk = active[start:end]
            vals = acq_fn(C[idx_chunk].unsqueeze(1)).view(-1)
            j_rel_chunk = int(vals.argmax().item())
            val_chunk = vals[j_rel_chunk]
            if (best_val is None) or (val_chunk > best_val):
                best_val = val_chunk
                best_j_rel = start + j_rel_chunk
        j = int(active[best_j_rel].item())
        selected_idx.append(j)
        if r_min is None:
            mask[j] = False
        else:
            dists = torch.cdist(C[active], C[j:j+1]).squeeze(-1)
            close = dists < float(r_min)
            mask[active[close]] = False
    return C[torch.tensor(selected_idx, device=device)]

def propose_points_from_plan(
    sampler_cls: Type,
    X: Tensor,
    y: Tensor,
    plan: dict,
    q: Optional[int] = None,
    *,
    num_paths_single: int = 512,
    num_paths_batch: int = 1024,
    cand_size: int = 8192,
    temperature: float = 1.0,
    min_dist: Optional[float] = None,
    distinct_paths: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    assert X.ndim == 2, "X shape should be [N, d]"
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    assert y.ndim == 2 and y.shape[0] == X.shape[0], "y shape should be [N, 1] and match X"
    N, d = X.shape
    device = device or (X.device if X.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    X = X.to(device=device, dtype=dtype).contiguous()
    y = y.to(device=device, dtype=dtype).contiguous()
    n_out = 1 if (q is None) else int(q)
    bounds = torch.stack([torch.zeros(d, dtype=dtype, device=device),
                          torch.ones(d, dtype=dtype, device=device)], dim=0)
    C = _make_sobol_grid_norm(d, cand_size, dtype, device)
    sampler = sampler_cls(bounds=bounds, X_init=X, Y_init=y, dtype=dtype, device=device)
    mode: str = plan.get("mode", "none")
    region_info = plan.get("region", None)
    x_star: Optional[Tensor] = plan.get("x_star", None)
    y_star = plan.get("y_star", None)
    delta = plan.get("delta", 0.6)
    smooth_override = (region_info or {}).get("smooth", None)
    if mode.startswith("region"):
        assert region_info is not None and {"lb","ub"} <= set(region_info.keys()), "region mode needs plan['region'] including lb/ub"
        lb: Tensor = region_info["lb"].to(device=device, dtype=dtype).clone().detach()
        ub: Tensor = region_info["ub"].to(device=device, dtype=dtype).clone().detach()
        grid_size = int(region_info.get("grid_size", 512))
        smooth = smooth_override
        smooth = 0.06 if smooth_override is None else float(smooth_override)
        pp = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds, grid_size=grid_size, smooth=smooth, dtype=dtype, device=device
        )
        pp.set_box_region(lb, ub)
        pp.fit_lambda_by_delta(base_model=sampler.model, delta=float(delta), observation_noise=False)
        pp.prepare_cache(base_model=sampler.model)
        effective = TiltedModel(sampler.model, pp).eval()
        y_best_std = sampler._current_best_std().item()
        mc_paths = num_paths_batch if n_out > 1 else num_paths_single
        acq = qLogExpectedImprovement(
            model=effective,
            best_f=y_best_std,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_paths]))
        )
        X_prop = _greedy_select_with_min_dist(acq, C, n=n_out, r_min=min_dist,
                                              chunk_size=128 if n_out > 1 else 256)
        return X_prop
    if mode == "value":
        assert y_star is not None, "value mode needs plan['y_star']"
        y_star_std = (torch.as_tensor(y_star, dtype=dtype, device=device) - sampler._y_mean) / sampler._y_std.clamp_min(1e-12)
        sampler.set_location_prior(
            ValuePeakPrior(
                y_star_std=float(y_star_std.item()),
                d=d, dtype=dtype, device=device,
                beta=float(delta),
                raw_samples=2**12
            )
        )
        return sampler.ask(
            n=n_out,
            num_paths=(num_paths_batch if n_out > 1 else num_paths_single),
            cand_size=cand_size,
            temperature=temperature,
            min_dist=min_dist,
            distinct_paths=distinct_paths,
        )
    if mode == "point":
        assert x_star is not None, "point mode needs plan['x_star']"
        sampler.set_location_prior(
            PointBumpPrior(x_star_norm=x_star.to(device=device, dtype=dtype), sigma=float(delta))
        )
        return sampler.ask(
            n=n_out,
            num_paths=(num_paths_batch if n_out > 1 else num_paths_single),
            cand_size=cand_size,
            temperature=temperature,
            min_dist=min_dist,
            distinct_paths=distinct_paths,
        )
    return sampler.ask(
        n=n_out,
        num_paths=(num_paths_batch if n_out > 1 else num_paths_single),
        cand_size=cand_size,
        temperature=temperature,
        min_dist=min_dist,
        distinct_paths=distinct_paths,
    )

def _mk_data(N=24, d=3, seed=0, device=None, dtype=torch.double):
    g = torch.Generator().manual_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    X = torch.rand(N, d, generator=g, dtype=dtype, device=device).clamp(1e-6, 1-1e-6)
    y = (
        -((X - 0.6)**2).sum(dim=-1, keepdim=True)
        + 0.8 * X[:, :1]
        + 0.1 * torch.randn(N, 1, generator=g, dtype=dtype, device=device)
    )
    return X, y, device, dtype

def _print_header(title: str):
    print("\n" + "=" * 12 + f" {title} " + "=" * 12)

def smoke_test_all(sampler_cls: Type, *, seed=0):
    X, y, device, dtype = _mk_data(N=48, d=3, seed=seed, dtype=torch.double)
    _print_header("REGION / REGION-SOFT")
    center = torch.tensor([0.5, 0.6, 0.6], dtype=dtype, device=device)
    radius = 0.10
    plan_region = decide_preference(
        kind="region",
        confidence=0.99,
        d=X.shape[1],
        region_center=center,
        region_radius=radius,
        grid_size=512,
        E_soft_low=2.0,
        E_soft_high=6.0,
        y_star=float(y.max().item()),
    )
    print("plan_region:", {k: (v if k != "region" else {kk: (vv if kk != "lb" and kk != "ub" else "Tensor[...]")
                                                        for kk, vv in v.items()})
                           for k, v in plan_region.items()})
    X1 = propose_points_from_plan(
        sampler_cls=sampler_cls,
        X=X, y=y,
        plan=plan_region,
        q=None,
        num_paths_single=512,
        cand_size=4096,
        temperature=1.0,
        min_dist=None,
        distinct_paths=True,
        device=device,
        dtype=dtype,
    )
    print("proposed (q=1):", X1)
    assert X1.shape == (1, X.shape[1])
    assert torch.all((X1 > 0) & (X1 < 1))
    Xk = propose_points_from_plan(
        sampler_cls=sampler_cls,
        X=X, y=y,
        plan=plan_region,
        q=5,
        num_paths_batch=1024,
        cand_size=8192,
        temperature=1.0,
        min_dist=0.05,
        distinct_paths=True,
        device=device,
        dtype=dtype,
    )
    print("proposed (q=5):", Xk)
    assert Xk.shape == (5, X.shape[1])
    assert torch.all((Xk > 0) & (Xk < 1))
    if plan_region["mode"].startswith("region"):
        lb = plan_region["region"]["lb"].to(device=device, dtype=dtype)
        ub = plan_region["region"]["ub"].to(device=device, dtype=dtype)
        in_box = ((Xk >= lb) & (Xk <= ub)).all(dim=-1).float().mean().item()
        print(f"in-box ratio (batch): {in_box:.2f}")
    _print_header("POINT")
    x_star = torch.tensor([0.8, 0.2, 0.4], dtype=dtype, device=device)
    plan_point = decide_preference(
        kind="point",
        confidence=0.80,
        d=X.shape[1],
        x_star=x_star,
        y_star=None,
    )
    print("plan_point:", plan_point)
    Xp = propose_points_from_plan(
        sampler_cls=sampler_cls,
        X=X, y=y,
        plan=plan_point,
        q=4,
        cand_size=4096,
        temperature=0.8,
        min_dist=0.05,
        device=device,
        dtype=dtype,
    )
    print("proposed (point, q=4):", Xp)
    assert Xp.shape == (4, X.shape[1])
    assert torch.all((Xp > 0) & (Xp < 1))
    _print_header("VALUE")
    y_star = float(y.max().item()) + 0.5
    plan_value = decide_preference(
        kind="value",
        confidence=0.95,
        d=X.shape[1],
        y_star=y_star,
    )
    print("plan_value:", plan_value)
    Xv = propose_points_from_plan(
        sampler_cls=sampler_cls,
        X=X, y=y,
        plan=plan_value,
        q=3,
        cand_size=4096,
        temperature=1.0,
        min_dist=0.05,
        device=device,
        dtype=dtype,
    )
    print("proposed (value, q=3):", Xv)
    assert Xv.shape == (3, X.shape[1])
    assert torch.all((Xv > 0) & (Xv < 1))
    print("\nAll smoke tests finished ✅")

def test_region_tilt_two_points(
    sampler_cls: Type,
    *,
    center: torch.Tensor,
    radius: float = 0.20,
    two_points: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    confidence: float = 0.80,
    smooth: Optional[float] = 0.06,
    sigma_floor: float = 5e-3,
    q: int = 5,
    cand_size: int = 8192,
    box_priority: bool = True,
    min_dist: Optional[float] = 0.05,
    seed: int = 0,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
):
    g = torch.Generator().manual_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    def synth(X: torch.Tensor) -> torch.Tensor:
        return (
            -((X - 0.6)**2).sum(dim=-1, keepdim=True)
            + 0.8 * X[:, :1]
        )
    if two_points is None:
        X_train = torch.rand(2, center.numel(), generator=g, dtype=dtype, device=device).clamp(1e-6, 1-1e-6)
    else:
        assert two_points.shape == (2, center.numel())
        X_train = two_points.to(device=device, dtype=dtype).contiguous()
    y_train = synth(X_train)
    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train, generator=g)
    d = center.numel()
    bounds = torch.stack([torch.zeros(d, dtype=dtype, device=device),
                          torch.ones(d,  dtype=dtype, device=device)], dim=0)
    C = _make_sobol_grid_norm(d, cand_size, dtype, device)
    lb = (center - radius).clamp(0, 1).to(device=device, dtype=dtype)
    ub = (center + radius).clamp(0, 1).to(device=device, dtype=dtype)
    sampler = sampler_cls(bounds=bounds, X_init=X_train, Y_init=y_train, dtype=dtype, device=device)
    p = float(min(max(confidence, 1e-12), 1 - 1e-12))
    z = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=dtype, device=device)))
    pp = LinearExponentialRegionalMeanTiltPlugAndPlay(
        bounds=bounds, grid_size=512, smooth=smooth, dtype=dtype, device=device
    )
    pp.set_box_region(lb, ub)
    pp.fit_lambda_by_delta(base_model=sampler.model, delta=float(z), observation_noise=False)
    pp.prepare_cache(base_model=sampler.model)
    effective = TiltedModel(sampler.model, pp).eval()
    post_g = sampler.model.posterior(pp.Xg, observation_noise=False)
    try:
        varS = float(post_g.mvn.lazy_covariance_matrix.quadratic_form(pp._a))
    except Exception:
        Sigma = post_g.mvn.covariance_matrix
        varS = float((pp._a * (Sigma @ pp._a)).sum())
    sigmaS = max(varS**0.5, float(sigma_floor))
    m0 = sampler.model.posterior(C).mean.squeeze(-1)
    m1 = effective.posterior(C).mean.squeeze(-1)
    mask_in = ((C >= lb) & (C <= ub)).all(dim=-1)
    uplift_in  = float((m1[mask_in]  - m0[mask_in]).mean()) if mask_in.any() else 0.0
    uplift_out = float((m1[~mask_in] - m0[~mask_in]).mean())
    print("\n============ REGION (2-point GP) ============")
    print(f"center={center.tolist()}, radius={radius:.3f}, smooth={None if smooth is None else float(smooth):.3f}")
    print(f"X_train={X_train.tolist()}")
    print(f"[debug] confidence={confidence:.3f}, z={z:.3f}, sigmaS={sigmaS:.3f}, lambda={float(pp._lam):.3f}")
    print(f"[debug] uplift_in≈{uplift_in:.3f}, uplift_out≈{uplift_out:.3f}")
    y_best_std = sampler._current_best_std().item()
    acq = qLogExpectedImprovement(
        model=effective, best_f=y_best_std,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
    )
    if box_priority:
        Cin, Cout = C[mask_in], C[~mask_in]
        need = q
        picked = []
        if Cin.numel() > 0:
            take_in = min(need, Cin.size(0))
            picked.append(_greedy_select_with_min_dist(acq, Cin, n=take_in, r_min=min_dist, chunk_size=128))
            need -= take_in
        if need > 0:
            picked.append(_greedy_select_with_min_dist(acq, Cout, n=need, r_min=min_dist, chunk_size=256))
        X_prop = torch.cat(picked, dim=0)
    else:
        X_prop = _greedy_select_with_min_dist(acq, C, n=q, r_min=min_dist, chunk_size=256)
    in_box_ratio = ((X_prop >= lb) & (X_prop <= ub)).all(dim=-1).float().mean().item()
    print(f"proposed (q={q}): {X_prop}")
    print(f"in-box ratio: {in_box_ratio:.2f}")
    acq_base = qLogExpectedImprovement(
        model=sampler.model, best_f=y_best_std,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
    )
    X_prop_base = _greedy_select_with_min_dist(acq_base, C, n=q, r_min=min_dist, chunk_size=256)
    in_box_ratio_base = ((X_prop_base >= lb) & (X_prop_base <= ub)).all(dim=-1).float().mean().item()
    print("---- baseline (no tilt) ----")
    print(f"proposed (q={q}): {X_prop_base}")
    print(f"in-box ratio: {in_box_ratio_base:.2f}")
    print("=============================================")

def make_multimodal_2d(dtype=torch.double, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    mus = torch.tensor([[0.80, 0.20],
                        [0.25, 0.75],
                        [0.50, 0.50]],
                       dtype=dtype, device=device)
    amps = torch.tensor([1.20, 1.00, 0.90], dtype=dtype, device=device)
    sigs = torch.tensor([0.04, 0.05, 0.07], dtype=dtype, device=device)
    def f(X: torch.Tensor) -> torch.Tensor:
        diffs = X.unsqueeze(1) - mus.unsqueeze(0)
        quad  = - (diffs**2).sum(dim=-1) / (2.0 * sigs)
        val   = torch.logsumexp(torch.log(amps) + quad, dim=-1, keepdim=True)
        val = val + 0.05 * X[:, :1] - 0.03 * X[:, 1:2]
        return val
    x_global_nominal = mus[0]
    y_global_nominal = f(x_global_nominal.unsqueeze(0)).item()
    def approx_global_max(n: int = 200000, seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        Xs = torch.rand(n, 2, generator=g, dtype=dtype, device=device)
        ys = f(Xs).squeeze(-1)
        j  = int(ys.argmax().item())
        return Xs[j], float(ys[j].item())
    return f, x_global_nominal, y_global_nominal, approx_global_max

def _plan_region(center, radius, confidence=0.95, grid_size=512, smooth=0.06):
    p = float(min(max(confidence, 1e-12), 1 - 1e-12))
    z = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0)))
    lb = (center - radius).clamp(0.0, 1.0)
    ub = (center + radius).clamp(0.0, 1.0)
    return {
        "mode": "region",
        "delta": z,
        "region": {"lb": lb, "ub": ub, "grid_size": grid_size, "smooth": smooth},
        "x_star": center,
        "y_star": None,
    }

def _plan_none():
    return {"mode": "none"}

def test_full_bo_with_region_prefs(
    sampler_cls: Type,
    *,
    T: int = 25,
    q: int = 1,
    N0: int = 6,
    noise_std: float = 0.01,
    confidence: float = 0.95,
    radius_local: float = 0.10,
    radius_global: float = 0.10,
    cand_size: int = 8192,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
    seed: int = 0,
):
    g = torch.Generator(device=device).manual_seed(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    f, x_global_nominal, y_global_nominal, approx_global_max = make_multimodal_2d(dtype=dtype, device=device)
    x_star_appx, y_star_appx = approx_global_max(n=200000, seed=seed)
    d = 2
    bounds = torch.stack([torch.zeros(d, dtype=dtype, device=device),
                          torch.ones(d,  dtype=dtype, device=device)], dim=0)
    X0 = torch.rand(N0, d, generator=g, dtype=dtype, device=device).clamp(1e-6, 1-1e-6)
    y0_true = f(X0)
    y0_obs  = y0_true + noise_std * torch.randn(N0, 1, generator=g, dtype=dtype, device=device)
    X_A, yA_fit, yA_true = X0.clone(), y0_obs.clone(), y0_true.clone()
    X_B, yB_fit, yB_true = X0.clone(), y0_obs.clone(), y0_true.clone()
    X_C, yC_fit, yC_true = X0.clone(), y0_obs.clone(), y0_true.clone()
    center_local  = torch.tensor([0.25, 0.75], dtype=dtype, device=device)
    center_global = torch.tensor([0.80, 0.20], dtype=dtype, device=device)
    plan_A = _plan_region(center_local,  radius_local,  confidence=confidence, smooth=0.06)
    plan_B = _plan_region(center_global, radius_global, confidence=confidence, smooth=0.06)
    plan_C = _plan_none()
    def _best_traj(y_hist_true: torch.Tensor):
        best, traj = -1e9, []
        for val in y_hist_true.squeeze(-1).tolist():
            best = max(best, val); traj.append(best)
        return traj
    best_traj_A = _best_traj(yA_true)
    best_traj_B = _best_traj(yB_true)
    best_traj_C = _best_traj(yC_true)
    for t in range(1, T + 1):
        for tag in ("A(local)", "B(global)", "C(none)"):
            if tag.startswith("A"):
                X, y_fit, y_true, plan = X_A, yA_fit, yA_true, plan_A
            elif tag.startswith("B"):
                X, y_fit, y_true, plan = X_B, yB_fit, yB_true, plan_B
            else:
                X, y_fit, y_true, plan = X_C, yC_fit, yC_true, plan_C
            sampler = sampler_cls(bounds=bounds, X_init=X, Y_init=y_fit, dtype=dtype, device=device)
            X_prop = propose_points_from_plan(
                sampler_cls=sampler_cls,
                X=X, y=y_fit, plan=plan, q=q,
                num_paths_single=512, num_paths_batch=1024,
                cand_size=cand_size, temperature=1.0,
                min_dist=None, distinct_paths=True,
                device=device, dtype=dtype,
            )
            y_prop_true = f(X_prop)
            y_prop_obs  = y_prop_true + noise_std * torch.randn(
                X_prop.size(0), 1, generator=g, dtype=dtype, device=device
            )
            X_new      = torch.cat([X,      X_prop], dim=0)
            y_fit_new  = torch.cat([y_fit,  y_prop_obs],  dim=0)
            y_true_new = torch.cat([y_true, y_prop_true], dim=0)
            if tag.startswith("A"):
                X_A, yA_fit, yA_true = X_new, y_fit_new, y_true_new
                best_traj_A = _best_traj(yA_true)
            elif tag.startswith("B"):
                X_B, yB_fit, yB_true = X_new, y_fit_new, y_true_new
                best_traj_B = _best_traj(yB_true)
            else:
                X_C, yC_fit, yC_true = X_new, y_fit_new, y_true_new
                best_traj_C = _best_traj(yC_true)
        if t % 5 == 0 or t == T:
            print(f"[iter {t:02d}] best (true) so far: "
                  f"A(local)={best_traj_A[-1]:.3f}, "
                  f"B(global)={best_traj_B[-1]:.3f}, "
                  f"C(none)={best_traj_C[-1]:.3f}")
    print("\n========= SUMMARY =========")
    print(f"Nominal global @ {x_global_nominal.tolist()} -> f ≈ {y_global_nominal:.3f}")
    print(f"[approx] true global ≈ f({x_star_appx.tolist()}) = {y_star_appx:.3f}")
    print(f"Final best (true)  A(local):  {best_traj_A[-1]:.3f}")
    print(f"Final best (true)  B(global): {best_traj_B[-1]:.3f}")
    print(f"Final best (true)  C(none):   {best_traj_C[-1]:.3f}")
    return {
        "A": {"X": X_A, "y_fit": yA_fit, "y_true": yA_true, "best_traj": best_traj_A, "plan": plan_A},
        "B": {"X": X_B, "y_fit": yB_fit, "y_true": yB_true, "best_traj": best_traj_B, "plan": plan_B},
        "C": {"X": X_C, "y_fit": yC_fit, "y_true": yC_true, "best_traj": best_traj_C, "plan": plan_C},
        "global_nominal": {"x": x_global_nominal, "y": y_global_nominal},
        "global_approx":  {"x": x_star_appx, "y": y_star_appx},
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = test_full_bo_with_region_prefs(
        sampler_cls=WeightedQLogEI,
        T=25, q=1, N0=2, noise_std=0.01,
        confidence=0.8,
        radius_local=0.10, radius_global=0.10,
        cand_size=8192,
        dtype=torch.double,
        device=device,
        seed=0,
    )
