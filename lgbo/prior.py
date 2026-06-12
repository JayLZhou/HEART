from __future__ import annotations

import os
import math
import warnings
import numpy as np
import torch
from torch import Tensor
import inspect
# 强烈建议：GP 用 double 提升稳定性
torch.set_default_dtype(torch.double)

# ===== BoTorch / GPyTorch =====
from botorch.test_functions import Branin
from botorch.utils.transforms import normalize, unnormalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.sampling.pathwise.posterior_samplers import (
    draw_matheron_paths,
    MatheronPath,
)
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.utils.sampling import manual_seed
from gpytorch.distributions import MultivariateNormal

# ====== 引入你提供的类（建议与本脚本放同目录或以模块方式导入） ======
# 为了示例直观，这里直接复制 import 路径；若你把类放到单独文件，请改为 from xxx import yyy
from typing import Optional, Sequence, Union

from botorch.utils.transforms import standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.models.model import Model

# ==== 这里粘贴你给出的类定义（为简洁省略，只需确保已经在同一命名空间中可用）====
# 你可直接把你上一条消息中的类定义粘贴到这里（UserPrior* / LinearExponentialRegionalMeanTiltPlugAndPlay / TiltedModel）
# ------------------ 开始粘贴你的类定义 ------------------
import torch.distributions as dist
from abc import ABC, abstractmethod
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch.sampling.pathwise import MatheronPath
from botorch.utils.sampling import optimize_posterior_samples
from gpytorch.distributions import MultivariateNormal

def make_matheron_paths(model, num_paths: int = 256, observation_noise: bool = False):
    """
    兼容不同 BoTorch 版本的 MatheronPath 构造器。
    优先使用 .from_gp；若不存在则回退到 draw_matheron_paths，再不行就报错提示版本过旧。
    """
    from botorch.sampling.pathwise import MatheronPath
    # 方案 A: 新接口
    if hasattr(MatheronPath, "from_gp"):
        return MatheronPath.from_gp(model, num_paths=num_paths, observation_noise=observation_noise)

    # 方案 B: 老接口，自己画 prior/update 两条路径
    try:
        # 有的版本放在这个路径
        from botorch.sampling.pathwise.paths import draw_matheron_paths
    except Exception:
        draw_matheron_paths = None

    if draw_matheron_paths is not None:
        prior_paths, update_paths = draw_matheron_paths(
            model, num_paths=num_paths, observation_noise=observation_noise
        )
        return MatheronPath(prior_paths=prior_paths, update_paths=update_paths)
# =========================== 基类：用户先验 ===========================
class UserPrior:
    pass


class UserPriorLocation(ABC):
    """
    用于“位置/argmax”类先验的基类：在输入空间上定义的先验。
    约定：
    - self.bounds 是原始空间 (2, d)
    - forward(X) 接收 **归一化**输入 X∈[0,1]^d，evaluate 返回 log-prob（(..., n, 1)）
    """

    def __init__(self, bounds: Tensor, prior_floor: float = 1e-12,
                 dtype: torch.dtype = torch.double, seed: int = 42):
        self.bounds = bounds
        self.norm_bounds = torch.tensor([(0.0, 1.0) for _ in range(bounds.shape[1])],
                                        dtype=dtype).T
        self.dim = bounds.shape[1]
        self.prior_floor = prior_floor
        self.dtype = dtype
        self.seed = seed
        self.optval_prior: Optional[UserPriorValue] = None

    def register_maxval(self, optval_prior: "UserPriorValue"):
        self.optval_prior = optval_prior

    def compute_logprobs(self, matheron_paths: MatheronPath,
                         raw_samples: int = 2 ** 11, **kwargs) -> Tensor:
        """
        给定函数样本（Matheron paths），返回每条路径“发生”的未归一化对数概率。
        """

        self.optimal_inputs, self.optimal_outputs = optimize_posterior_samples(
            matheron_paths,
            bounds=self.norm_bounds,         # (2,d) in [0,1]
            raw_samples=raw_samples,         # 仍可用
            num_restarts=20,                  # 允许为 0
        )
        logprobs = self.forward(self.optimal_inputs)
        if self.optval_prior is not None:
            logprobs_output = self.optval_prior.evaluate(self.optimal_outputs)
            logprobs = logprobs + logprobs_output
        return logprobs

    def get_optima(self):
        return self.optimal_inputs, self.optimal_outputs

    def compute_norm_probs(
        self,
        matheron_paths: MatheronPath,
        decay_factor: Optional[Union[Tensor, int, float]] = 1.0,
        prior_floor: Optional[Union[Tensor, int, float]] = 0.0,
        **kwargs,
    ) -> Tensor:
        """
        将 log 概率做稳定化、幂次衰减与归一化，得到“相对概率”分配。
        """
        logprobs = self.compute_logprobs(matheron_paths, **kwargs)
        logprobs_norm = logprobs - logprobs.max()
        probs = torch.exp(logprobs_norm)
        decay_probs = torch.pow(probs, decay_factor).clamp_min(prior_floor)
        N_paths = probs.shape[-2]          # (num_optima, num_paths, out_dim) 中的路径维
        norm_probs = (N_paths * decay_probs) / decay_probs.sum()
        return norm_probs

    @abstractmethod
    def evaluate(self, X: Tensor) -> Tensor:
        """在 **归一化**输入 X 上返回 (..., n, 1) 形状的 log 概率密度"""
        pass

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # evaluate 返回 (..., n, 1)；这里 squeeze(1) 对齐你原有语义
        return self.evaluate(X).squeeze(1)

    @abstractmethod
    def sample(self, num_samples: int = 1) -> Tensor:
        """从 **原始空间**采样（上层 _sample 会 normalize）"""
        pass

    def _sample(self, num_samples: int = 1) -> Tensor:
        """从 **归一化**空间采样（外层逻辑通常需要这个版本）"""
        return normalize(self.sample(num_samples=num_samples), self.bounds)


# =========================== 具体：默认值高斯先验 ===========================
class DefaultPrior(UserPriorLocation):
    """
    基于参数默认值的独立高斯先验（在 **归一化**空间上定义）。
    confidence：标量或 (d,)，表示在 [0,1] 空间中的标准差。
    """

    def __init__(self, bounds: Tensor, parameter_defaults: Tensor,
                 confidence: Optional[float] = 0.25, spread_dim: bool = True):
        super().__init__(bounds)
        self.dim = bounds.shape[1]
        # 归一化的默认值 μ
        self.parameter_defaults = normalize(parameter_defaults, bounds)
        assert self.parameter_defaults.dim() == 1 and self.parameter_defaults.shape[0] == self.dim, \
            f"parameter_defaults 应为 shape=({self.dim},) 的 1D 向量"

        # 归一化尺度下的 std
        if isinstance(confidence, (float, int)):
            confidence = torch.full((self.dim,), float(confidence),
                                    dtype=self.parameter_defaults.dtype,
                                    device=self.parameter_defaults.device)
        else:
            confidence = torch.as_tensor(confidence,
                                         dtype=self.parameter_defaults.dtype,
                                         device=self.parameter_defaults.device)
            assert confidence.shape == (self.dim,), "confidence 需为标量或长度为 d 的张量"

        self.priors_list = []
        self.norm_factors = []
        for mu, std in zip(self.parameter_defaults, confidence):
            std = torch.clamp(std, min=1e-6)
            distr = dist.Normal(mu, std)
            # 截断到 [0,1] 的归一化常数
            Z = distr.cdf(torch.tensor(1.0, dtype=mu.dtype, device=mu.device)) - \
                distr.cdf(torch.tensor(0.0, dtype=mu.dtype, device=mu.device))
            self.priors_list.append(distr)
            self.norm_factors.append(Z)

    @property
    def default(self) -> Tensor:
        # 原始空间默认值
        return unnormalize(self.parameter_defaults, self.bounds)

    @property
    def _default(self) -> Tensor:
        # **归一化**默认值（用于拼候选）
        return self.parameter_defaults

    def sample(self, num_samples: int) -> Tensor:
        # 在 **归一化**空间采样，再反归一化回原始空间
        out_norm = torch.empty(num_samples, self.dim,
                               dtype=self.parameter_defaults.dtype,
                               device=self.parameter_defaults.device)
        for d in range(self.dim):
            out_norm[:, d] = self.priors_list[d].rsample(torch.Size([num_samples]))
        return unnormalize(out_norm, self.bounds)

    def evaluate(self, X: Tensor) -> Tensor:
        # X: (..., n, d)  -> 返回 (..., n, 1) 的 log 概率密度
        log_prob = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)  # (..., n)
        for d in range(self.dim):
            lp = self.priors_list[d].log_prob(X[..., d])
            Z = torch.clamp(self.norm_factors[d],
                            min=torch.tensor(1e-12, dtype=lp.dtype, device=lp.device))
            lp = lp - torch.log(Z)  # 截断校正
            log_prob = log_prob + lp
        return log_prob.unsqueeze(-1)


# =========================== 具体：偏好型先验（better vs worse） ===========================
class PreferencePrior(UserPriorLocation):
    """
    偏好先验：给定“更好”的配置 better_configs 与“更差”的配置 worse_configs，
    二者都应位于 **归一化**空间 [0,1]^d。
    打分：越靠近 better 越高，越靠近 worse 越低（返回 log-score）。
    """

    def __init__(self, bounds: Tensor, better_configs: Tensor, worse_configs: Tensor):
        super().__init__(bounds)
        self.dim = bounds.shape[1]
        self.better_configs = better_configs.clone()
        self.worse_configs = worse_configs.clone()
        assert self.better_configs.dim() == 2 and self.better_configs.shape[1] == self.dim
        assert self.worse_configs.dim() == 2 and self.worse_configs.shape[1] == self.dim

    @property
    def _default(self) -> Tensor:
        # 用于 compute_logprobs 周边扰动：返回 **归一化**默认向量
        return self.better_configs[0]

    def sample(self, num_samples: int) -> Tensor:
        if len(self.better_configs) < num_samples:
            raise ValueError(
                f"PreferencePrior: better_configs 只有 {len(self.better_configs)} 条，无法无放回采样 {num_samples} 条"
            )
        idx = np.random.choice(len(self.better_configs), size=num_samples, replace=False)
        # 外层 _sample() 会再 normalize，这里需要返回 **原始空间** 样本
        return unnormalize(self.better_configs[idx], self.bounds)

    def evaluate(self, X: Tensor) -> Tensor:
        # X: (..., n, d) in [0,1]^d；返回 (..., n, 1) 的 log-score
        eps = 1e-12
        alpha = 50.0
        beta = 50.0

        orig_shape = X.shape[:-1]  # (..., n)
        X2 = X.reshape(-1, self.dim)  # (N, d)

        def min_sq_dist(A: Tensor, B: Tensor) -> Tensor:
            d2 = (A[:, None, :] - B[None, :, :]).pow(2).sum(-1)  # (N, M)
            return d2.min(dim=1).values  # (N,)

        d2_better = min_sq_dist(X2, self.better_configs)
        d2_worse = min_sq_dist(X2, self.worse_configs)

        s_num = torch.exp(-alpha * d2_better)
        s_den = torch.exp(-beta * d2_worse)
        score = torch.log(s_num + eps) - torch.log(s_den + eps)
        return score.reshape(*orig_shape, 1)


# =========================== 具体：最优值先验（Y*） ===========================
class UserPriorValue(UserPrior, ABC):
    def __init__(self, prior_floor: float = 1e-12,
                 dtype: torch.dtype = torch.double, seed: int = 42):
        self.prior_floor = prior_floor
        self.dtype = dtype
        self.seed = seed
        self.mean = None
        self.std = None

    def setup(self, Y_normalized: Tensor, mean: Tensor, std: Tensor):
        self.Y_unnormalized = Y_normalized * std + mean
        self.mean = mean
        self.std = std

    def _unnormalize(self, Y: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            return Y
        return Y * self.std + self.mean

    @abstractmethod
    def evaluate(self, Y_opt: Tensor) -> Tensor:
        pass

    @t_batch_mode_transform()
    def forward(self, Y: Tensor) -> Tensor:
        return self.evaluate(Y)


class UserPriorHardMaxValue(UserPriorValue):
    def __init__(self, maxopt_value: Optional[float] = None, minopt_value: Optional[float] = None,
                 prior_floor: float = 1e-12, dtype: torch.dtype = torch.double, seed: int = 42):
        super().__init__(prior_floor=prior_floor, dtype=dtype, seed=seed)
        self.minopt_value = minopt_value
        self.maxopt_value = maxopt_value

    def evaluate(self, Y_opt: Tensor) -> Tensor:
        Y_unnormalized = self._unnormalize(Y_opt)
        mask = torch.ones_like(Y_unnormalized, dtype=torch.bool)
        if self.minopt_value is not None:
            mask = mask & (Y_unnormalized > self.minopt_value)
        if self.maxopt_value is not None:
            mask = mask & (Y_unnormalized < self.maxopt_value)
        # 返回 log(prob)：True->0，False->-inf（用 prior_floor 截断）
        return torch.log(mask.to(Y_opt.dtype) + self.prior_floor)


class UserPriorMaxValue(UserPriorValue):
    def __init__(self, parameter_default: float, confidence: float,
                 prior_floor: float = 1e-12, dtype: torch.dtype = torch.double, seed: int = 42):
        super().__init__(prior_floor=prior_floor, dtype=dtype, seed=seed)
        self.prior_dist = dist.Normal(
            torch.tensor([parameter_default], dtype=dtype),
            torch.tensor([max(confidence, 1e-6)], dtype=dtype),
        )

    def evaluate(self, Y_opt: Tensor) -> Tensor:
        Y_unnormalized = self._unnormalize(Y_opt)
        logp = self.prior_dist.log_prob(Y_unnormalized)
        return torch.clamp(logp, min=math.log(self.prior_floor))


# =========================== 区域均值平移：开箱即用 ===========================
def _make_sobol_grid_norm(dim: int, n_points: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    eng = torch.quasirandom.SobolEngine(dim)
    X = eng.draw(n_points).to(dtype=dtype, device=device)
    eps = torch.tensor(1e-6, dtype=dtype, device=device)
    return X.clamp(eps, 1 - eps)

def _box_mask_norm(X: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    return ((X >= lb) & (X <= ub)).all(dim=-1)


class LinearExponentialRegionalMeanTiltPlugAndPlay: 
    r"""
    线性指数型区域先验（解析均值平移）：
      ρ(f) = exp( λ * Σ_g a_g f(x_g) )
    - 在 [0,1]^d 的 Sobol 网格上构造权重 a（区域内均匀，或平滑）
    - 通过 δ 标定 λ：λ = δ / (a^T Σ_F a)
    - 倾斜后：μ_ρ(X) = μ(X) + λ * Σ_{X,G} a，协方差不变
    """

    def __init__(
        self,
        bounds: Tensor,                  # (2, d) 原始边界
        grid_size: int = 512,
        smooth: Optional[float] = None,  # 若给定，用 sigmoid 软边界
        dtype: torch.dtype = torch.double,
        device: Optional[torch.device] = None,

        
    ):
        assert bounds.dim() == 2 and bounds.shape[0] == 2
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.dtype = dtype
        self.device = device or bounds.device

        self.Xg = _make_sobol_grid_norm(self.d, grid_size, dtype, self.device)  # (G,d)
        self.smooth = smooth
        self._a: Optional[Tensor] = None  # (G,)
        self._lam: Optional[float] = None

        self._region_lb: Optional[Tensor] = None
        self._region_ub: Optional[Tensor] = None

    # ---------- 区域设定 ----------
    def set_box_region(self, lb_norm: Union[Sequence[float], Tensor],
                       ub_norm: Union[Sequence[float], Tensor]) -> None:
        lb = torch.as_tensor(lb_norm, dtype=self.dtype, device=self.device)
        ub = torch.as_tensor(ub_norm, dtype=self.dtype, device=self.device)
        assert lb.shape == (self.d,) and ub.shape == (self.d,)
        self._region_lb, self._region_ub = lb.clamp(0, 1), ub.clamp(0, 1)
        self._build_a()

    def set_dim_interval(self, dim: int, lo: float, hi: float) -> None:
        assert 0 <= dim < self.d
        lb = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        ub = torch.ones(self.d, dtype=self.dtype, device=self.device)
        lo, hi = float(lo), float(hi)
        if hi < lo:
            lo, hi = hi, lo
        lb[dim], ub[dim] = max(0.0, lo), min(1.0, hi)
        self._region_lb, self._region_ub = lb, ub
        self._build_a()

    def _build_a(self) -> None:
        assert self._region_lb is not None and self._region_ub is not None, "请先设置区域"
        if self.smooth is None:
            mask = _box_mask_norm(self.Xg, self._region_lb, self._region_ub)  # (G,)
            a_raw = mask.to(self.dtype)
        else:
            s = torch.tensor(self.smooth, dtype=self.dtype, device=self.device)
            z1 = (self.Xg - self._region_lb) / (s + 1e-12)
            z2 = (self._region_ub - self.Xg) / (s + 1e-12)
            soft = torch.sigmoid(z1).prod(dim=-1) * torch.sigmoid(z2).prod(dim=-1)  # (G,)
            a_raw = soft
        ssum = torch.clamp(a_raw.sum(), min=1e-12)
        self._a = (a_raw / ssum).to(self.dtype)

    # ---------- 强度设定 ----------
    def set_lambda(self, lam: float) -> None:
        self._lam = float(lam)

    @torch.no_grad()
    def fit_lambda_by_delta(self, base_model: Model, delta: float,
                            observation_noise: bool = False, **posterior_kwargs) -> None:
        """
        通过“区域期望增量 δ”标定 λ：λ = δ / (a^T Σ_F a)
        Σ_F 只在 Sobol 网格 Xg 上涉及，不需要全矩阵，用 quadratic_form 更快。
        """
        assert self._a is not None, "请先设置区域 a"
        post = base_model.posterior(
            self.Xg, observation_noise=observation_noise, **posterior_kwargs
        )
        mvn = post.mvn
        a = self._a

        try:
            # 优先用 lazy 结构，直接算二次型（O(G)）
            Sigma_F_lazy = mvn.lazy_covariance_matrix
            aSa = Sigma_F_lazy.quadratic_form(a)  # shape=()
        except (AttributeError, NotImplementedError):
            # 回退：显式 covariance_matrix（O(G^2)）
            try:
                Sigma_F = mvn.covariance_matrix
            except AttributeError:
                Sigma_F = mvn.lazy_covariance_matrix.evaluate()

            Sa = Sigma_F @ a
            aSa = (a * Sa).sum()

        denom = float(max(aSa.item() if torch.is_tensor(aSa) else aSa, 1e-6))
        denom = math.sqrt(denom)
        self._lam = float(delta) / denom
        
    def _ensure_region(self) -> None:
        assert self._a is not None, "请先设置区域 a"

    @torch.no_grad()
    def prepare_cache(self, base_model: Model, observation_noise: bool = False) -> None:
        """
        预计算与训练集有关的量：A^{-1} K(X_tr, Xg) a
        只依赖训练数据与 a，候选 X 不参与，因此可复用。
        """
        self._ensure_region()
        # 训练输入
        Xtr = base_model.train_inputs[0]
        dtype = self.dtype
        device = self.device

        # K_tt + σ^2 I
        K_tt = base_model.covar_module(Xtr, Xtr).evaluate()
        noise = base_model.likelihood.noise if hasattr(base_model.likelihood, "noise") else torch.tensor(0.0, dtype=dtype, device=device)
        A = K_tt + noise * torch.eye(K_tt.size(-1), dtype=K_tt.dtype, device=K_tt.device)

        # K_tg a
        K_tg = base_model.covar_module(Xtr, self.Xg).evaluate()          # [n_tr, G]
        v = K_tg @ self._a                                               # [n_tr]

        # u = A^{-1} v
        L = torch.linalg.cholesky(A)                                     # [n_tr, n_tr]
        u = torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1)         # [n_tr]

        # 缓存
        self._cache = {"Xtr": Xtr, "L": L, "u": u}
    # ---------- 倾斜后验（只平移均值，协方差不变） ----------
    @torch.no_grad()
    def posterior(self, base_model: Model, X: Tensor, observation_noise: bool = False,
                  **posterior_kwargs) -> GPyTorchPosterior:
        self._ensure_region()
        assert self._lam is not None, "请先设定 λ（set_lambda 或 fit_lambda_by_delta）"

        # # 基础后验（拿 μ_X 与 Σ_XX；这里不触碰 Xg，避免大联合）
        # post_X = base_model.posterior(X, observation_noise=observation_noise, **posterior_kwargs)
        # mvn_X = post_X.mvn
        # try:
        #     cov_XX = mvn_X.covariance_matrix
        # except AttributeError:
        #     cov_XX = mvn_X.lazy_covariance_matrix.evaluate()
        # mean_X = mvn_X.mean  # [n]
        post_X = base_model.posterior(X, observation_noise=observation_noise, **posterior_kwargs)
        mvn_X  = post_X.mvn
        mean_X = mvn_X.mean  # [n]

        # 1) 保留 Lazy 协方差，并显式加 jitter
        cov_lazy = mvn_X.lazy_covariance_matrix
        cov_lazy = cov_lazy.add_jitter(1e-6)   # 如仍报错可调到 1e-5 或 1e-4

        # 如果有 cache：用 K_xg a - K_xt u 的轻量公式
        cache = getattr(self, "_cache", None)
        if cache is not None:
            Xtr, L, u = cache["Xtr"], cache["L"], cache["u"]
            K_xg_a = base_model.covar_module(X, self.Xg).evaluate() @ self._a     # [n]
            K_xt   = base_model.covar_module(X, Xtr).evaluate()                    # [n, n_tr]
            K_xt_u = K_xt @ u                                                      # [n]
            shift  = K_xg_a - K_xt_u                                               # [n]
        else:
            # 退路（兼容老逻辑）：拼联合再切片 —— 可能内存重，不推荐大 M 时走这里
            X_cat = torch.cat([X, self.Xg.expand(*X.shape[:-2], *self.Xg.shape)], dim=-2)
            post_joint = base_model.posterior(X_cat, observation_noise=observation_noise, **posterior_kwargs)
            mvn = post_joint.mvn
            try:
                cov_full = mvn.covariance_matrix
            except AttributeError:
                cov_full = mvn.lazy_covariance_matrix.evaluate()
            n = X.shape[-2]; G = self.Xg.shape[-2]
            cov_XG = cov_full[..., :n, n:n+G]
            shift  = torch.matmul(cov_XG, self._a.view(-1,1)).squeeze(-1)

        # mean_tilt = mean_X + self._lam * shift
        # mvn_tilt  = MultivariateNormal(mean_tilt, cov_XX)
        # return GPyTorchPosterior(mvn_tilt)
        mean_tilt = mean_X + self._lam * shift

        # 2) 用 Lazy 协方差来构造 MVN（不要 evaluate 成稠密）
        mvn_tilt  = MultivariateNormal(mean_tilt, cov_lazy)
        return GPyTorchPosterior(mvn_tilt)
class TiltedModel(Model):
    """
    轻量封装：把“区域抬升”作用在任意 base_model 上，直接喂给 EI/NEI 等采集函数。
    - 提供 num_outputs / subset_output
    - 兼容某些代码访问 `basemodel`
    - 安全的 __getattr__，避免递归
    """
    def __init__(self, base_model: Model, mean_tilt_pp: LinearExponentialRegionalMeanTiltPlugAndPlay):
        super().__init__()
        # 先直接用 object.__setattr__，确保属性已存在，避免 __getattr__ 在构造期被触发
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "mean_tilt_pp", mean_tilt_pp)

    # ---- 兼容某些代码访问 `model.basemodel` ----
    @property
    def basemodel(self) -> Model:
        return object.__getattribute__(self, "base_model")

    # ---- 采集函数常用到的接口 ----
    @property
    def num_outputs(self) -> int:
        bm = object.__getattribute__(self, "base_model")
        return getattr(bm, "num_outputs", 1)

    def subset_output(self, idcs: Tensor) -> "TiltedModel":
        bm = object.__getattribute__(self, "base_model")
        if hasattr(bm, "subset_output"):
            sub = bm.subset_output(idcs)
            return TiltedModel(sub, object.__getattribute__(self, "mean_tilt_pp"))
        return self  # 单输出时通常不会被调用

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        bm = object.__getattribute__(self, "base_model")
        pp = object.__getattribute__(self, "mean_tilt_pp")
        return pp.posterior(bm, X, observation_noise=observation_noise, **kwargs)

    # ---- 常见方法直接转发，保持状态一致 ----
    def train(self, mode: bool = True):
        bm = object.__getattribute__(self, "base_model")
        bm.train(mode)
        return self

    def eval(self):
        bm = object.__getattribute__(self, "base_model")
        bm.eval()
        return self

    def to(self, *args, **kwargs):
        bm = object.__getattribute__(self, "base_model")
        bm = bm.to(*args, **kwargs)
        object.__setattr__(self, "base_model", bm)
        return self

    # ---- 安全转发的 __getattr__（只在本类没有该属性时触发）----
    def __getattr__(self, name):
        # 避免递归：不要在这里再用 getattr(self, "base_model")
        if name in ("base_model", "mean_tilt_pp"):
            raise AttributeError(name)
        try:
            bm = object.__getattribute__(self, "base_model")
        except AttributeError:
            # 构造期间或异常状态，直接抛出
            raise AttributeError(name)
        return getattr(bm, name)
    
# def make_branin_data(n: int = 16, seed: int = 0):
#     tf = Branin(negate=True)  # 我们最大化（所以取负号）
#     bounds = tf.bounds  # (2, d)
#     d = bounds.shape[1]
#     with manual_seed(seed):
#         X = torch.rand(n, d, dtype=torch.double)
#     Y = tf(unnormalize(X, bounds)).unsqueeze(-1)
#     return tf, bounds, X, Y


# def fit_stgp(X: Tensor, Y: Tensor) -> SingleTaskGP:
#     Ystd = standardize(Y)
#     model = SingleTaskGP(train_X=X, train_Y=Ystd)
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     fit_gpytorch_mll(mll)
#     model.eval()
#     return model


# def test_location_priors_and_matheron(model: Model, bounds: Tensor):
#     # -------- 小工具：Spearman 相关（纯 torch 实现） --------
#     def _spearman(x: Tensor, y: Tensor) -> float:
#         # rank via double argsort
#         rx = torch.argsort(torch.argsort(x))
#         ry = torch.argsort(torch.argsort(y))
#         rx = rx.to(torch.double)
#         ry = ry.to(torch.double)
#         # 皮尔逊相关 on ranks
#         rx = (rx - rx.mean()) / (rx.std() + 1e-12)
#         ry = (ry - ry.mean()) / (ry.std() + 1e-12)
#         return float((rx * ry).mean().item())

#     def _warn(msg: str):
#         warnings.warn(msg, RuntimeWarning)

#     torch.manual_seed(0); np.random.seed(0)

#     print("\n==== [Test 1] DefaultPrior + MatheronPath ====")
#     d = bounds.shape[1]
#     defaults = unnormalize(torch.full((d,), 0.5, dtype=torch.double), bounds)
#     prior = DefaultPrior(bounds=bounds, parameter_defaults=defaults, confidence=0.2)

#     # 构造 MatheronPath（使用你的模型即可）
#     paths = draw_matheron_paths(
#         model,
#         sample_shape=torch.Size([256]),   # 指定路径条数
#     )

#     # ---------- Test 1：位置先验 ----------
#     logps = prior.compute_logprobs(paths, raw_samples=2**9)
#     probs = prior.compute_norm_probs(paths, decay_factor=1.0)
#     print(f"logps shape={tuple(logps.shape)} | probs shape={tuple(probs.shape)}")

#     # 硬正确性（assert）
#     assert probs.shape == logps.shape, "probs / logps 形状不一致"
#     assert logps.shape[-1] == 1, "当前实现只支持单输出先验打分（out_dim 应为 1）"
#     assert torch.isfinite(logps).all(), "logps 出现 NaN/Inf"
#     # 注意：compute_norm_probs 返回与 logps 同形；这里验证总和≈元素个数
#     total = probs.sum()
#     expected = torch.tensor(float(probs.numel() // probs.shape[-1]))  # 等价 len(flattened paths)
#     # 由于 probs 是三维 (1,P,1)，numel()/last_dim == P；与老语义“和=长度”一致z``
#     assert torch.allclose(total, expected, rtol=1e-5, atol=1e-6), "归一化失败：sum != N_paths"

#     # 软效果（报告 + 告警）
#     xopt = prior.optimal_inputs.reshape(-1, d)            # [P, d]
#     logps1d = logps.squeeze(-1).reshape(-1)               # [P]
#     # 默认点（归一化空间）的距离
#     x_def_norm = prior._default
#     dist = torch.linalg.norm(xopt - x_def_norm, dim=-1)   # [P]
#     rho = _spearman(dist, logps1d)
#     print(f"[Test1] spearman(dist, logps): {rho:+.3f}  (更近应更高 -> 期望为负)")
#     if not (rho < -0.30):
#         _warn(f"[Test1] 相关性偏弱（{rho:+.3f} ≥ -0.30），可能是采样稀疏或维度较高造成的波动")

#     # ---------- Test 2：叠加值先验 ----------
#     print("\n==== [Test 2] + UserPriorMaxValue (value prior) ====")
#     ymean = model.posterior(torch.rand(8, d)).mean.mean()  # 用后验均值粗略设一个“目标值”
#     val_prior = UserPriorMaxValue(parameter_default=float(ymean.item()), confidence=1.0)
#     prior.register_maxval(val_prior)

#     logps2 = prior.compute_logprobs(paths, raw_samples=2**9)
#     print(f"with value prior, logps shape={tuple(logps2.shape)}")
#     assert logps2.shape == logps.shape, "叠加值先验后形状应与位置先验一致"
#     assert torch.isfinite(logps2).all(), "logps2 出现 NaN/Inf"

#     logps2_1d = logps2.squeeze(-1).reshape(-1)
#     gain = logps2_1d - logps1d  # 按路径对应的增量
#     fopt = prior.optimal_outputs.squeeze(-1).reshape(-1)  # [P]，与 logps2 对齐
#     delta = (fopt - ymean).abs()
#     rho_val = _spearman(delta, logps2_1d)
#     print(f"[Test2] spearman(|f*-target|, logps_val): {rho_val:+.3f}  (越近越高 -> 期望为负)")
#     if not (rho_val < -0.30):
#         _warn(f"[Test2] 值先验方向性偏弱（{rho_val:+.3f} ≥ -0.30）")

#     # 近/远两组的增量比较（用中位数阈）
#     med = torch.median(delta)
#     gain_near = gain[delta <= med].mean().item()
#     gain_far  = gain[delta >  med].mean().item()
#     print(f"[Test2] mean gain (near - far): {gain_near - gain_far:+.4f}")
#     if not ((gain_near - gain_far) > 0.0):
#         _warn("[Test2] 近目标组的增益未高于远组，建议增大 raw_samples 或检查值先验参数")

#     # ---------- Test 3：偏好先验 ----------
#     print("\n==== [Test 3] PreferencePrior ====")
#     better = torch.tensor([[0.1, 0.7], [0.2, 0.6]], dtype=torch.double)[:, :d]
#     worse  = torch.tensor([[0.8, 0.2], [0.9, 0.3]], dtype=torch.double)[:, :d]
#     pref = PreferencePrior(bounds=bounds, better_configs=better, worse_configs=worse)
#     logps_pref = pref.compute_logprobs(paths, raw_samples=2**9)
#     print(
#         f"Preference logps shape={tuple(logps_pref.shape)}; "
#         f"mean={float(logps_pref.mean()):+.4f}, std={float(logps_pref.std()):.4f}"
#     )
#     assert logps_pref.shape == logps.shape, "偏好先验形状不一致"
#     assert torch.isfinite(logps_pref).all(), "logps_pref 出现 NaN/Inf"

#     # 效果度量：靠近 better 且远离 worse 的 margin 越大，分数应越高
#     xopt_pref = pref.optimal_inputs.reshape(-1, d)
#     logps_pref_1d = logps_pref.squeeze(-1).reshape(-1)

#     def _min_sq_dist(A: Tensor, B: Tensor) -> Tensor:
#         # (N,d) vs (M,d) -> N
#         return ((A[:, None, :] - B[None, :, :])**2).sum(-1).min(dim=1).values

#     d_b = torch.sqrt(_min_sq_dist(xopt_pref, better))
#     d_w = torch.sqrt(_min_sq_dist(xopt_pref, worse))
#     margin = d_w - d_b  # 越大越偏向 better
#     rho_pref = _spearman(margin, logps_pref_1d)
#     print(f"[Test3] spearman(margin, logps): {rho_pref:+.3f}  (越偏向 better 越高 -> 期望为正)")
#     if not (rho_pref > +0.30):
#         _warn(f"[Test3] 偏好先验方向性偏弱（{rho_pref:+.3f} ≤ +0.30）")

# def test_tilt_model(model: Model, bounds: Tensor):
#     print("\n==== [Test 4] LinearExponentialRegionalMeanTiltPlugAndPlay + TiltedModel ====")

#     d = bounds.shape[1]
#     pp = LinearExponentialRegionalMeanTiltPlugAndPlay(bounds=bounds, grid_size=256, smooth=0.06)

#     # 例如：只在第 0 维高区间抬升
#     pp.set_dim_interval(dim=0, lo=0.7, hi=1.0)

#     # 通过 delta 标定 lambda（区域均值增量）
#     pp.fit_lambda_by_delta(base_model=model, delta=1.0, observation_noise=False)

#     tilted = TiltedModel(model, pp)

#     # 随机取两批点：inside（x0 >= 0.75）与 outside（x0 <= 0.25）
#     N = 32
#     X_all = torch.rand(N, d, dtype=torch.double)
#     inside_mask = X_all[:, 0] >= 0.75
#     outside_mask = X_all[:, 0] <= 0.25
#     X_in = X_all[inside_mask]
#     X_out = X_all[outside_mask]

#     if len(X_in) < 4 or len(X_out) < 4:  # 保底再造一些
#         X_in = torch.rand(32, d, dtype=torch.double)
#         X_in[:, 0] = 0.85
#         X_out = torch.rand(32, d, dtype=torch.double)
#         X_out[:, 0] = 0.15

#     # 比较抬升前后的均值
#     post_in_base = model.posterior(X_in).mean
#     post_in_tilt = tilted.posterior(X_in).mean
#     post_out_base = model.posterior(X_out).mean
#     post_out_tilt = tilted.posterior(X_out).mean

#     lift_in = (post_in_tilt - post_in_base).mean().item()
#     lift_out = (post_out_tilt - post_out_base).mean().item()

#     print(f"Inside-region mean lift ≈ {lift_in:+.4f} | Outside-region mean lift ≈ {lift_out:+.4f}")

#     # 断言：区域内的平均抬升应显著高于区域外
#     assert lift_in > lift_out - 1e-6, "区域内抬升不明显，请检查 a / λ 或协方差抓取是否正确"

#     # 协方差不变（抽样比较一个点）
#     X_probe = torch.rand(6, d, dtype=torch.double)
#     cov_base = model.posterior(X_probe).mvn.covariance_matrix
#     cov_tilt = tilted.posterior(X_probe).mvn.covariance_matrix
#     diff_cov = (cov_tilt - cov_base).abs().max().item()

#     print(f"Max |Cov_tilt - Cov_base| = {diff_cov:.2e}")
#     assert diff_cov < 1e-8, "倾斜后协方差应不变"

# def main():
#     warnings.filterwarnings("ignore")
#     torch.manual_seed(0)
#     np.random.seed(0)

#     # 1) 构造训练数据并拟合 GP
#     tf, bounds, X, Y = make_branin_data(n=24, seed=0)
#     model = fit_stgp(X, Y)

#     # 2) 仅测试“别人的先验”实现与效果（去掉你的区域平移测试）
#     test_location_priors_and_matheron(model, bounds)

#     test_tilt_model(model, bounds)
#     print("\n✅ Tests finished (hard checks passed). See warnings for soft-effect diagnostics.")

# if __name__ == "__main__": 
#     main()