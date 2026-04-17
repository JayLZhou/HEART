from __future__ import annotations

import math
from itertools import islice
from typing import Any, Dict, Sequence

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from Tuner.BOTuner.lgbo_components.history import LGBOObservation
from Tuner.BOTuner.lgbo_components.search_space import MixedSearchSpaceAdapter, ParamSpec


def _make_sobol_grid_norm(dim: int, n_points: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    engine = torch.quasirandom.SobolEngine(dim, scramble=True)
    grid = engine.draw(n_points).to(dtype=dtype, device=device)
    eps = torch.tensor(1e-6, dtype=dtype, device=device)
    return grid.clamp(eps, 1 - eps)


def _norm_ppf(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.double)))


@torch.no_grad()
def _greedy_select_with_min_dist(acq_fn, candidates: Tensor, n: int, r_min: float | None, *, chunk_size: int = 256) -> Tensor:
    count = candidates.size(0)
    device = candidates.device
    mask = torch.ones(count, dtype=torch.bool, device=device)
    selected_idx = []
    for _ in range(n):
        active = torch.arange(count, device=device)[mask]
        if active.numel() == 0:
            break
        best_val = None
        best_j_rel = None
        for start in range(0, active.numel(), chunk_size):
            end = min(start + chunk_size, active.numel())
            idx_chunk = active[start:end]
            vals = acq_fn(candidates[idx_chunk].unsqueeze(1)).view(-1)
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
            dists = torch.cdist(candidates[active], candidates[j : j + 1]).squeeze(-1)
            close = dists < float(r_min)
            mask[active[close]] = False
    return candidates[torch.tensor(selected_idx, device=device)]


class LinearExponentialRegionalMeanTiltPlugAndPlay:
    """Region-lift posterior tilt adapted from the reference LGBO code."""

    def __init__(
        self,
        bounds: Tensor,
        *,
        grid_size: int = 256,
        smooth: float | None = None,
        dtype: torch.dtype = torch.double,
        device: torch.device | None = None,
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape [2, d]")
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.dtype = dtype
        self.device = device or bounds.device
        self.Xg = _make_sobol_grid_norm(self.d, grid_size, dtype, self.device)
        self.smooth = smooth
        self._a: Tensor | None = None
        self._lam: float | None = None
        self._region_lb: Tensor | None = None
        self._region_ub: Tensor | None = None

    def set_box_region(self, lb_norm: Sequence[float] | Tensor, ub_norm: Sequence[float] | Tensor) -> None:
        lb = torch.as_tensor(lb_norm, dtype=self.dtype, device=self.device)
        ub = torch.as_tensor(ub_norm, dtype=self.dtype, device=self.device)
        if lb.shape != (self.d,) or ub.shape != (self.d,):
            raise ValueError("region bounds must match dimensionality")
        self._region_lb = lb.clamp(0, 1)
        self._region_ub = ub.clamp(0, 1)
        self._build_a()

    def _build_a(self) -> None:
        if self._region_lb is None or self._region_ub is None:
            raise ValueError("region must be set before computing weights")
        if self.smooth is None:
            mask = ((self.Xg >= self._region_lb) & (self.Xg <= self._region_ub)).all(dim=-1)
            a_raw = mask.to(self.dtype)
        else:
            smooth = torch.tensor(self.smooth, dtype=self.dtype, device=self.device)
            z1 = (self.Xg - self._region_lb) / (smooth + 1e-12)
            z2 = (self._region_ub - self.Xg) / (smooth + 1e-12)
            a_raw = torch.sigmoid(z1).prod(dim=-1) * torch.sigmoid(z2).prod(dim=-1)
        total = torch.clamp(a_raw.sum(), min=1e-12)
        self._a = (a_raw / total).to(self.dtype)

    @torch.no_grad()
    def fit_lambda_by_delta(self, base_model: Model, delta: float, observation_noise: bool = False) -> None:
        if self._a is None:
            raise ValueError("region weights are not initialized")
        post = base_model.posterior(self.Xg, observation_noise=observation_noise)
        mvn = post.mvn
        try:
            covariance = mvn.lazy_covariance_matrix
            a_sa = covariance.quadratic_form(self._a)
        except (AttributeError, NotImplementedError):
            try:
                dense = mvn.covariance_matrix
            except AttributeError:
                dense = mvn.lazy_covariance_matrix.to_dense()
            a_sa = (self._a * (dense @ self._a)).sum()
        denom = float(max(a_sa.item() if torch.is_tensor(a_sa) else a_sa, 1e-6))
        self._lam = float(delta) / math.sqrt(denom)

    @torch.no_grad()
    def prepare_cache(self, base_model: Model) -> None:
        if self._a is None:
            raise ValueError("region weights are not initialized")
        train_x = base_model.train_inputs[0]
        k_tt = base_model.covar_module(train_x, train_x).evaluate()
        noise = getattr(base_model.likelihood, "noise", torch.tensor(0.0, dtype=self.dtype, device=self.device))
        chol = torch.linalg.cholesky(k_tt + noise * torch.eye(k_tt.size(-1), dtype=k_tt.dtype, device=k_tt.device))
        k_tg = base_model.covar_module(train_x, self.Xg).evaluate()
        u = torch.cholesky_solve((k_tg @ self._a).unsqueeze(-1), chol).squeeze(-1)
        self._cache = {"train_x": train_x, "u": u}

    @torch.no_grad()
    def posterior(
        self,
        base_model: Model,
        X: Tensor,
        observation_noise: bool = False,
        **posterior_kwargs,
    ) -> GPyTorchPosterior:
        if self._a is None or self._lam is None:
            raise ValueError("tilt must be configured before posterior evaluation")
        post_x = base_model.posterior(X, observation_noise=observation_noise, **posterior_kwargs)
        mvn_x = post_x.mvn
        mean_x = mvn_x.mean
        cov_lazy = mvn_x.lazy_covariance_matrix.add_jitter(1e-6)
        cache = getattr(self, "_cache", None)
        if cache is not None:
            train_x = cache["train_x"]
            u = cache["u"]
            k_xg_a = base_model.covar_module(X, self.Xg).evaluate() @ self._a
            k_xt = base_model.covar_module(X, train_x).evaluate()
            shift = k_xg_a - (k_xt @ u)
        else:
            x_cat = torch.cat([X, self.Xg.expand(*X.shape[:-2], *self.Xg.shape)], dim=-2)
            post_joint = base_model.posterior(x_cat, observation_noise=observation_noise, **posterior_kwargs)
            mvn_joint = post_joint.mvn
            try:
                cov_full = mvn_joint.covariance_matrix
            except AttributeError:
                cov_full = mvn_joint.lazy_covariance_matrix.to_dense()
            n = X.shape[-2]
            g = self.Xg.shape[-2]
            cov_xg = cov_full[..., :n, n : n + g]
            shift = torch.matmul(cov_xg, self._a.view(-1, 1)).squeeze(-1)
        mean_tilt = mean_x + self._lam * shift
        return GPyTorchPosterior(MultivariateNormal(mean_tilt, cov_lazy))


class TiltedModel(Model):
    def __init__(self, base_model: Model, mean_tilt_pp: LinearExponentialRegionalMeanTiltPlugAndPlay) -> None:
        super().__init__()
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "mean_tilt_pp", mean_tilt_pp)

    @property
    def basemodel(self) -> Model:
        return object.__getattribute__(self, "base_model")

    @property
    def num_outputs(self) -> int:
        return getattr(object.__getattribute__(self, "base_model"), "num_outputs", 1)

    def subset_output(self, idcs: Tensor) -> "TiltedModel":
        base_model = object.__getattribute__(self, "base_model")
        if hasattr(base_model, "subset_output"):
            return TiltedModel(base_model.subset_output(idcs), object.__getattribute__(self, "mean_tilt_pp"))
        return self

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        return object.__getattribute__(self, "mean_tilt_pp").posterior(
            object.__getattribute__(self, "base_model"),
            X,
            observation_noise=observation_noise,
            **kwargs,
        )

    def __getattr__(self, name: str):
        if name in {"base_model", "mean_tilt_pp"}:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "base_model"), name)


class LGBOMixedBayesGenerator:
    """Fit a mixed surrogate over encoded HEART parameters and optimize qLogEI."""

    def __init__(
        self,
        *,
        candidate_pool_size: int = 8192,
        grid_size: int = 512,
        mc_paths: int = 512,
        min_dist: float | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        self.candidate_pool_size = candidate_pool_size
        self.grid_size = grid_size
        self.mc_paths = mc_paths
        self.min_dist = min_dist
        self.dtype = dtype
        self.space = MixedSearchSpaceAdapter()
        self.max_category_assignments = 256

    def propose(
        self,
        *,
        plan: Dict[str, Any] | None,
        observations: Sequence[LGBOObservation],
        specs: Sequence[ParamSpec],
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        if len(observations) < 2:
            raise ValueError("Need at least two completed observations for surrogate fitting")

        device = torch.device("cpu")
        train_x = []
        train_y = []
        for obs in observations:
            train_x.append(self.space.encode_point(obs.params, specs))
            objective = float(obs.objective)
            train_y.append(objective if higher_is_better else -objective)

        X = torch.tensor(train_x, dtype=self.dtype, device=device)
        Y = torch.tensor(train_y, dtype=self.dtype, device=device).unsqueeze(-1)
        Y_std, _, _ = self._standardize(Y)

        bounds = torch.stack(
            [
                torch.zeros(self.space.encoded_dim(specs), dtype=self.dtype, device=device),
                torch.ones(self.space.encoded_dim(specs), dtype=self.dtype, device=device),
            ],
            dim=0,
        )
        model = SingleTaskGP(train_X=X, train_Y=Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        effective_model = self._build_effective_model(model=model, plan=plan, specs=specs, bounds=bounds)
        acquisition = qLogExpectedImprovement(
            model=effective_model,
            best_f=Y_std.max(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_paths])),
        )
        acq_fn = acquisition
        if plan and plan.get("mode") == "point":
            x_star_point = plan.get("x_star") or plan.get("point")
            if x_star_point:
                x_star = torch.tensor(
                    self.space.encode_point(x_star_point, specs),
                    dtype=self.dtype,
                    device=device,
                )
                sigma = max(1e-6, float(plan.get("delta", 0.6)))

                def _acq_with_point_prior(Xq: Tensor) -> Tensor:
                    vals = acquisition(Xq).view(-1)
                    Xflat = Xq[:, 0, :] if Xq.ndim == 3 else Xq
                    sqdist = ((Xflat - x_star.unsqueeze(0)) ** 2).sum(dim=-1)
                    log_prior = -0.5 * sqdist / (sigma * sigma)
                    return vals + log_prior

                acq_fn = _acq_with_point_prior
        plan_for_candidates = plan
        # Align with lgbo/boo.py point mode behavior: point prior guides
        # acquisition, but sampling remains global (no hard single-point clip).
        if plan and plan.get("mode") == "point":
            plan_for_candidates = None
        candidate_points = self._candidate_points(specs, plan=plan_for_candidates)
        encoded_candidates = torch.tensor(
            [self.space.encode_point(point, specs) for point in candidate_points],
            dtype=self.dtype,
            device=device,
        )
        proposed = _greedy_select_with_min_dist(acq_fn, encoded_candidates, n=1, r_min=self.min_dist)
        selected = proposed[0]
        best_idx = int(torch.argmin(torch.cdist(encoded_candidates, selected.unsqueeze(0))).item())
        return self.space.clip_point(candidate_points[best_idx], specs)

    def _build_effective_model(
        self,
        *,
        model: Model,
        plan: Dict[str, Any] | None,
        specs: Sequence[ParamSpec],
        bounds: Tensor,
    ) -> Model:
        if not plan:
            return model

        mode = plan.get("mode")
        if mode not in {"region", "region-soft"}:
            return model
        lower = plan["lower"]
        upper = plan["upper"]
        region_info = plan.get("region", {}) if isinstance(plan.get("region"), dict) else {}
        smooth_override = region_info.get("smooth")
        if smooth_override is None:
            smooth = 0.08 if mode == "region-soft" else None
        else:
            smooth = float(smooth_override)
        grid_size = int(region_info.get("grid_size", self.grid_size))

        lower_enc, upper_enc = self.space.encode_region_bounds(
            {"lower": lower, "upper": upper},
            specs,
        )
        lb = torch.tensor(lower_enc, dtype=self.dtype, device=bounds.device)
        ub = torch.tensor(upper_enc, dtype=self.dtype, device=bounds.device)
        tilt = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds,
            grid_size=grid_size,
            smooth=smooth,
            dtype=self.dtype,
            device=bounds.device,
        )
        tilt.set_box_region(lb, ub)
        delta = float(plan.get("delta", _norm_ppf(float(plan.get("confidence", 0.5)))))
        tilt.fit_lambda_by_delta(
            base_model=model,
            delta=delta,
            observation_noise=False,
        )
        tilt.prepare_cache(model)
        return TiltedModel(model, tilt).eval()

    def _standardize(self, Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean = Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            torch.tensor(1e-12, dtype=Y.dtype, device=Y.device)
        )
        return (Y - mean) / std, mean, std

    def _candidate_points(self, specs: Sequence[ParamSpec], plan: Dict[str, Any] | None = None) -> list[Dict[str, Any]]:
        categorical_assignments = self.space.enumerate_categorical_assignments(specs)
        if len(categorical_assignments) > self.max_category_assignments:
            categorical_assignments = list(islice(categorical_assignments, self.max_category_assignments))

        numeric_specs = self.space.numeric_specs(specs)
        if numeric_specs:
            numeric_grid = _make_sobol_grid_norm(
                len(numeric_specs),
                max(1, self.candidate_pool_size // max(1, len(categorical_assignments))),
                self.dtype,
                torch.device("cpu"),
            )
            numeric_points = []
            for row in numeric_grid:
                numeric_points.append(
                    self.space.denormalize_numeric_point(
                        {
                            spec.name: float(row[idx].item())
                            for idx, spec in enumerate(numeric_specs)
                        },
                        specs,
                    )
                )
        else:
            numeric_points = [{}]

        candidates: list[Dict[str, Any]] = []
        for cat in categorical_assignments:
            for num in numeric_points:
                point = self.space.clip_point(self.space.merge_points(num, cat), specs)
                if self.space.point_in_region(point, plan, specs):
                    candidates.append(point)
        if not candidates:
            fallback = self.space.default_point(specs)
            if plan:
                fallback = self.space.clip_point_to_region(fallback, plan, specs)
            return [fallback]
        return candidates
