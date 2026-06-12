from __future__ import annotations

import torch
from torch import Tensor
from abc import ABC, abstractmethod

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement
# 路径采样（0.15.1 用这个）
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths, MatheronPath
from prior import (
    _make_sobol_grid_norm,
    _box_mask_norm,
    LinearExponentialRegionalMeanTiltPlugAndPlay,
    TiltedModel,
)

# 你已有的先验类型
# - UserPriorLocation / UserPriorValue / DefaultPrior / PreferencePrior ...
#   这里仅用到接口：location_prior.compute_norm_probs(paths, ...)
#   若设置了 value_prior，请确保已经 register 到 location_prior 中。
#   e.g. location_prior.register_maxval(value_prior)

class BaseBOSampler(ABC):
    """
    通用 BO 基类（不指定采样/采集策略）。
    - 管理数据：X/Y（默认 X 为归一化 [0,1]^d，Y 可做标准化喂入 GP）
    - 拟合/更新 GP 模型（SingleTaskGP，标准化 Y）
    - 生成 Matheron 路径（用于基于路径的策略 or 先验加权）
    - 计算先验权重（若未设置先验则返回均匀权重）
    - 把“如何选点”的逻辑留给子类: `propose(...)`

    形状约定：
    - 路径调用与权重：采用 (num_optima, num_paths, out_dim) 的 3D 约定
      通常为 (1, P, 1)
    """

    def __init__(
        self,
        bounds: Tensor,                 # (2, d) 原始空间边界
        X_init: Tensor,                 # (n, d) 归一化到 [0,1]^d 的训练输入
        Y_init: Tensor,                 # (n, 1) 训练输出（原始量纲；内部会 standardize）
        dtype: torch.dtype = torch.double,
        device: torch.device | None = None,
    ):
        assert bounds.shape == (2, X_init.shape[-1])
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.dtype = dtype
        self.device = device or X_init.device

        # 数据：内部约定 X 为归一化；Y 为原始量纲（另存标准化后的）
        self.train_X = X_init.to(dtype=dtype, device=self.device)
        self.train_Y = Y_init.to(dtype=dtype, device=self.device)

        # 记录 Y 的标准化参数（供需要时反归一或设置 value-prior）
        self._y_mean: Tensor | None = None
        self._y_std: Tensor | None = None

        # 先验（可选）
        self.location_prior = None       
        self.value_prior = None         

        # 模型
        self.model: SingleTaskGP | None = None
        self.fit_model()  # 初始化拟合一次

    # ---------- 公共接口 ----------

    def set_location_prior(self, prior_loc) -> None:
        """设置位置型先验（Default / Preference 等）"""
        self.location_prior = prior_loc
        # 若已有值先验，注册到位置先验上
        if self.value_prior is not None and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)

    def set_value_prior(self, prior_val) -> None:
        """设置最优值先验（MaxValue/HardMaxValue）"""
        self.value_prior = prior_val
        # 注册到位置先验（若已存在）
        if self.location_prior is not None and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)
        # 若你希望值先验在“原始量纲”下工作，可在此处设置 mean/std
        # 例： prior_val.mean = self._y_mean ; prior_val.std = self._y_std

    def ask(
        self,
        n: int = 1,
        num_paths: int = 256,
        **kwargs,
    ) -> Tensor:
        """
        产生候选点（归一化 [0,1]^d）。
        - 生成路径
        - 计算先验权重（若无先验则均匀）
        - 调用子类实现的 propose(...) 来选点
        返回： (n, d) in [0,1]^d
        """
        assert self.model is not None, "model 尚未拟合"
        paths = self._make_paths(num_paths=num_paths)
        weights = self._compute_prior_weights(paths, **kwargs)  # (1, P, 1) 或同形
        X_next = self.propose(paths=paths, weights=weights, n=n, **kwargs)
        # 形状与范围保护
        X_next = X_next.to(dtype=self.dtype, device=self.device)
        eps = torch.tensor(1e-6, dtype=self.dtype, device=self.device)
        return X_next.clamp(eps, 1 - eps)

    def tell(self, X_new: Tensor, Y_new: Tensor, refit: bool = True) -> None:
        """
        追加新数据（X_new 需为归一化 [0,1]^d；Y_new 为原始量纲）。
        """
        X_new = X_new.to(dtype=self.dtype, device=self.device)
        Y_new = Y_new.to(dtype=self.dtype, device=self.device)
        self.train_X = torch.cat([self.train_X, X_new], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y_new], dim=0)
        if refit:
            self.fit_model()

    # ---------- 抽象：由子类实现选点策略 ----------

    @abstractmethod
    def propose(
        self,
        paths: MatheronPath,
        weights: Tensor,
        n: int = 1,
        **kwargs,
    ) -> Tensor:
        """
        子类实现“如何选点”的策略，返回 (n, d) in [0,1]^d。
        - 可以用 acquisition（EI/NEI/UCB）
        - 可以做 Thompson / 路径重采样
        - 可以混合先验权重等
        """
        raise NotImplementedError

    # ---------- 内部工具 ----------

    def fit_model(self) -> None:
        """用标准化 Y 拟合/更新 SingleTaskGP；记录 y 的均值/方差。"""
        Ystd, mean, std = self._standardize_with_stats(self.train_Y)
        self._y_mean, self._y_std = mean, std
        model = SingleTaskGP(train_X=self.train_X, train_Y=Ystd)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        self.model = model

    @torch.no_grad()
    def _make_paths(self, num_paths: int = 256, observation_noise: bool = False) -> MatheronPath:
        """从当前模型生成 Matheron 路径（0.15.1：直接返回 MatheronPath；无 observation_noise 参数）"""
        assert self.model is not None
        paths = draw_matheron_paths(
            self.model,
            sample_shape=torch.Size([num_paths]),
        )  # -> MatheronPath
        return paths

    def _compute_prior_weights(
        self,
        paths: MatheronPath,
        raw_samples: int = 2**10,
        decay_factor: float = 1.0,
        prior_floor: float = 0.0,
        **kwargs,
    ) -> Tensor:
        """
        计算/返回先验权重（默认均匀）。保持与 log-score 同形：(1, P, 1)。
        注意：若调用方需要 1D，可在外部 squeeze/reshape。
        """
        if self.location_prior is None:
            # 均匀权重（和 = P）
            P = paths.sample_shape[0] if hasattr(paths, "sample_shape") else None
            # 兼容性保险：若拿不到 P，就从一次假输入推断
            if P is None:
                Xprobe = torch.rand(3, self.d, dtype=self.dtype, device=self.device)
                Yprobe = paths(Xprobe.unsqueeze(-3))  # (P, 3, 1)
                P = Yprobe.shape[0]
            w = torch.ones(1, P, 1, dtype=self.dtype, device=self.device)
            return (P * w) / w.sum()

        # 若设置了值先验，确保已注册
        if (self.value_prior is not None) and hasattr(self.location_prior, "register_maxval"):
            self.location_prior.register_maxval(self.value_prior)

        # 交给你已有的先验实现（它应返回与 logps 同形）
        w = self.location_prior.compute_norm_probs(
            paths,
            decay_factor=decay_factor,
            prior_floor=prior_floor,
            raw_samples=raw_samples,
            **kwargs,
        )
        # 保险：再归一一次，确保和 = num_paths
        P = w.shape[-2]
        return (P * w) / (w.sum() + 1e-12)

    # ---------- 静态/小工具 ----------
    def _current_best_std(self) -> torch.Tensor:
        assert self._y_mean is not None and self._y_std is not None, "Call fit_model() first."
        Ystd = (self.train_Y - self._y_mean) / self._y_std.clamp_min(
            torch.tensor(1e-12, dtype=self._y_std.dtype, device=self._y_std.device)
        )
        return Ystd.max()
    @staticmethod
    def _standardize_with_stats(Y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """返回 (Ystd, mean, std)，std 做 >=1e-12 截断。"""
        mean = Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, unbiased=False, keepdim=True).clamp_min(torch.tensor(1e-12, dtype=Y.dtype, device=Y.device))
        return (Y - mean) / std, mean, std

    def sobol_candidates(self, n: int) -> Tensor:
        """在 [0,1]^d 采样 n 个 Sobol 候选（归一化空间）。"""
        eng = torch.quasirandom.SobolEngine(self.d, scramble=True)
        X = eng.draw(n).to(dtype=self.dtype, device=self.device)
        eps = torch.tensor(1e-6, dtype=self.dtype, device=self.device)
        return X.clamp(eps, 1 - eps)

    def to_raw(self, X_norm: Tensor) -> Tensor:
        """把归一化 X 转回原始空间。"""
        return unnormalize(X_norm, self.bounds)

    def to_norm(self, X_raw: Tensor) -> Tensor:
        """把原始空间 X 归一化到 [0,1]^d。"""
        return normalize(X_raw, self.bounds)


class WeightedQEI(BaseBOSampler):
    """
    加权 Monte Carlo qEI：
    EI(x) ≈ sum_i w_i * max(f_i(x) - y_best, 0)
    批量 q>1：贪心地每次选一个，使 “增量 qEI” 最大。
    """
    def propose(
        self,
        paths,
        weights: Tensor,      # (1,P,1) ; 和= P
        n: int = 1,
        cand_size: int = 8192,
        temperature: float = 1.0,   # 权重退火，<1 更均衡，>1 更尖锐
        **kwargs,
    ) -> Tensor:
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)       # [P]
        w = (w ** temperature)
        w = w / (w.sum() + 1e-12)                # 归一化为 1

        # 候选集（[0,1]^d）
        C = self.sobol_candidates(cand_size)     # [M,d]
        # 评估所有路径在候选的值（标准化标尺）
        Y = paths(C.unsqueeze(-3)).squeeze(-1).squeeze(1)   # [P, M]

        # 当前最好（标准化）：和 GP 训练标尺一致
        y_best = self._current_best_std()        # 标量 Tensor

        # 单点改进：I_i(x) = relu(Y - y_best)
        relu = torch.nn.ReLU()
        I = relu(Y - y_best)                     # [P, M]

        if n == 1:
            score = (w.view(-1,1) * I).sum(dim=0)    # [M]
            j = int(score.argmax().item())
            return C[j:j+1]

        # q>1：贪心，每步挑最大“增量 EI”
        selected = []
        # 每条路径的“当前批内最大值”初始化为 -inf
        cur = torch.full((P,), -1e30, dtype=Y.dtype, device=Y.device)
        base_gain = relu(cur - y_best)           # [P], 初始 0

        M = C.shape[0]
        mask = torch.ones(M, dtype=torch.bool, device=C.device)

        for _ in range(n):
            # 对每个候选，若加入它，批内每条路径最大值变为 max(cur, Y[:,j])
            # 增量改进：delta_ij = relu(max(cur, Y[:,j]) - y_best) - relu(cur - y_best)
            Yj = Y[:, mask]                      # [P, M_active]
            new_max = torch.maximum(cur.view(-1,1), Yj)
            delta = relu(new_max - y_best) - base_gain.view(-1,1)  # [P, M_active]
            score = (w.view(-1,1) * delta).sum(dim=0)               # [M_active]
            rel_idx = int(score.argmax().item())

            # 映射回全局索引 j
            j = torch.arange(M, device=mask.device)[mask][rel_idx].item()
            selected.append(j)

            # 更新路径内最大值与基线增益
            cur = torch.maximum(cur, Y[:, j])
            base_gain = relu(cur - y_best)
            mask[j] = False

        Xnext = C[torch.tensor(selected, device=C.device)]
        return Xnext


class WeightedQLogEI(BaseBOSampler):
    """
    加权 Monte Carlo qLogEI：
    logEI(x) ≈ sum_i w_i * log(1 + max(f_i(x) - y_best, 0))
    批量 q>1：同样用贪心增量，但把增量函数替换成 log1p(relu(...)) 的差分。
    """
    def propose(
        self,
        paths,
        weights: Tensor,
        n: int = 1,
        cand_size: int = 2048,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tensor:
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)       # [P]
        w = (w ** temperature)
        w = w / (w.sum() + 1e-12)

        C = self.sobol_candidates(cand_size)     # [M,d]
        Y = paths(C.unsqueeze(-3)).squeeze(-1).squeeze(1)   # [P, M]
        y_best = self._current_best_std()
        relu = torch.nn.ReLU()

        if n == 1:
            I = torch.log1p(relu(Y - y_best))    # [P, M]
            score = (w.view(-1,1) * I).sum(dim=0)
            j = int(score.argmax().item())
            return C[j:j+1]

        selected = []
        cur = torch.full((P,), -1e30, dtype=Y.dtype, device=Y.device)
        base_gain = torch.log1p(relu(cur - y_best))  # [P]
        M = C.shape[0]
        mask = torch.ones(M, dtype=torch.bool, device=C.device)

        for _ in range(n):
            Yj = Y[:, mask]                      # [P, M_active]
            new_max = torch.maximum(cur.view(-1,1), Yj)
            delta = torch.log1p(relu(new_max - y_best)) - base_gain.view(-1,1)
            score = (w.view(-1,1) * delta).sum(dim=0)
            rel_idx = int(score.argmax().item())
            j = torch.arange(M, device=mask.device)[mask][rel_idx].item()

            selected.append(j)
            cur = torch.maximum(cur, Y[:, j])
            base_gain = torch.log1p(relu(cur - y_best))
            mask[j] = False

        Xnext = C[torch.tensor(selected, device=C.device)]
        return Xnext


class WeightedTS(BaseBOSampler):
    """
    Weighted Thompson Sampling:
    - Sample a path index i ~ Categorical(w).
    - On that path, pick argmax over a candidate set.
    - For batch n>1, optionally use distinct paths and also avoid duplicate candidates.
    """
    def propose(
        self,
        paths: MatheronPath,
        weights: Tensor,               # shape (1, P, 1); sum = P
        n: int = 1,
        cand_size: int = 2048,
        temperature: float = 1.0,
        distinct_paths: bool = True,   # enforce different paths when n>1
        **kwargs,
    ) -> Tensor:
        # ----- prepare weights -----
        P = weights.shape[-2]
        w = weights.squeeze(0).squeeze(-1)       # [P]
        # temperature sharpening / smoothing
        w = (w ** temperature)
        w = w / (w.sum() + 1e-12)                # normalize to 1

        # ----- draw candidate pool and evaluate paths -----
        # C in [0,1]^d
        C = self.sobol_candidates(cand_size)     # [M, d]
        with torch.no_grad():
            # Y: [P, M] on standardized scale (consistent with model training)
            Y = paths(C.unsqueeze(-3)).squeeze(-1).squeeze(1)

        # ----- single point -----
        if n == 1:
            # single draw on path distribution
            i = torch.multinomial(w, 1, replacement=True).item()
            j = int(Y[i].argmax().item())
            return C[j:j+1]

        # ----- batch points (n > 1) -----
        M = C.shape[0]
        Xs = []

        # track used candidates to avoid duplicates
        used = torch.zeros(M, dtype=torch.bool, device=C.device)

        # working copy of weights so we can zero-out picked paths (if needed)
        w_work = w.clone()

        for t in range(n):
            # choose whether we can reuse a path
            if distinct_paths:
                replacement = (t >= P)  # after P picks, allow reuse
            else:
                replacement = True

            # sample a path index using current working weights
            # (if all weights are zero due to previous picks, reset)
            if w_work.sum() <= 1e-12:
                w_work = w.clone()
            i = torch.multinomial(w_work, 1, replacement=replacement).item()

            # pick argmax on candidate values for this path, avoiding used candidates
            # mask-out used candidates by setting them to a very negative value
            y_i = Y[i].clone()
            if (~used).any():
                y_i[used] = -1e30
                j = int(y_i.argmax().item())
            else:
                # fallback: if all candidates are used (n > M), allow duplicates
                j = int(Y[i].argmax().item())

            Xs.append(C[j])
            used[j] = True

            # if we enforce distinct paths and did not allow replacement,
            # zero-out the selected path's weight and renormalize
            if distinct_paths and not replacement:
                w_work[i] = 0.0
                s = w_work.sum()
                if s > 1e-12:
                    w_work = w_work / s
                else:
                    # degenerate safeguard: reset to original distribution
                    w_work = w.clone()

        return torch.stack(Xs, dim=0)

#==================================test code========================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch
# from torch import Tensor

# # ====== 这里假设三类已在同一文件中定义 ======
# # 如果你是分文件，请改成:
# # from your_file import WeightedQEI, WeightedQLogEI, WeightedTS, BaseBOSampler
# from typing import Type

# ---------------- 人工目标函数（在 [0,1]^d 上） ----------------
def toy_objective(X_norm: Tensor) -> Tensor:
    """
    X_norm: [n, d] in [0,1]
    返回 Y: [n, 1]，多峰函数 + 轻微噪声（默认无噪声，可改 noise_std）
    """
    noise_std = 0.00
    d = X_norm.shape[-1]
    # 两个高斯峰 + 一个正弦-余弦项
    mu1 = torch.full((d,), 0.25, dtype=X_norm.dtype, device=X_norm.device)
    mu2 = torch.full((d,), 0.75, dtype=X_norm.dtype, device=X_norm.device)
    s1 = 0.04
    s2 = 0.03
    g1 = torch.exp(-((X_norm - mu1) ** 2).sum(dim=-1) / (2 * s1**2))
    g2 = torch.exp(-((X_norm - mu2) ** 2).sum(dim=-1) / (2 * s2**2))
    trig = 0.2 * (torch.sin(6.28318 * X_norm).prod(dim=-1) + torch.cos(6.28318 * X_norm).mean(dim=-1))
    y = 1.5 * g1 + 2.0 * g2 + trig
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return y.unsqueeze(-1)  # [n,1]

# # ----------------- 通用：跑一个采样器若干步 -----------------
# def run_one_sampler(
#     sampler_cls: Type,
#     d: int = 3,
#     n_init: int = 12,
#     iters: int = 3,
#     batch: int = 3,
#     *,
#     # 超参数入口（仅通过本函数控制，不改采样器类）
#     num_paths_single: int = 512,   # 单点阶段的 P
#     num_paths_batch: int = 1024,   # 批量阶段的 P
#     cand_size: int = 8192,         # 候选池 M（传给各类 propose）
#     temperature: float = 1.0,      # 权重退火（传给各类 propose）
#     min_dist: float | None = None, # 批量多样性最小间距（支持则生效，否则被忽略）
#     distinct_paths: bool = True,   # TS 专属；其他策略会忽略
#     seed: int = 2025,
#     sobol_seed: int = 42,
#     device: torch.device | None = None,
#     dtype: torch.dtype = torch.double,
# ):
#     torch.manual_seed(seed)
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 边界（原始空间，这里与归一化相同）
#     bounds = torch.stack(
#         [
#             torch.zeros(d, dtype=dtype, device=device),
#             torch.ones(d, dtype=dtype, device=device),
#         ],
#         dim=0,
#     )  # [2,d]

#     # 初始点：Sobol
#     eng = torch.quasirandom.SobolEngine(d, scramble=True, seed=sobol_seed)
#     X0 = eng.draw(n_init).to(dtype=dtype, device=device)  # [n_init, d] in [0,1]
#     Y0 = toy_objective(X0)                                 # [n_init, 1]

#     # 实例化采样器
#     sampler = sampler_cls(bounds=bounds, X_init=X0, Y_init=Y0, dtype=dtype, device=device)

#     print(
#         f"\n==== {sampler_cls.__name__} | d={d}, init={n_init}, iters={iters} ====\n"
#         f"[cfg] P(single)={num_paths_single}, P(batch)={num_paths_batch}, "
#         f"M={cand_size}, temp={temperature}, min_dist={min_dist}, distinct_paths={distinct_paths}"
#     )

#     def report(tag: str):
#         ybest = sampler.train_Y.max().item()
#         print(f"{tag} | current best (raw scale) = {ybest:.4f} | n_obs = {sampler.train_X.shape[0]}")

#     report("Start")

#     # -------- 单点 ask/tell --------
#     X1 = sampler.ask(
#         n=1,
#         num_paths=num_paths_single,
#         cand_size=cand_size,
#         temperature=temperature,
#         min_dist=min_dist,          # 若策略未实现，会被 **kwargs 吞掉
#         distinct_paths=distinct_paths,
#     )
#     assert torch.all((0.0 <= X1) & (X1 <= 1.0)), "ask(n=1) 应返回 [0,1]^d 内的点"
#     Y1 = toy_objective(X1)
#     sampler.tell(X1, Y1, refit=True)
#     report("After 1-point ask/tell")

#     # -------- 批量 ask/tell（检查无重复）--------
#     Xq = sampler.ask(
#         n=batch,
#         num_paths=num_paths_batch,
#         cand_size=cand_size,
#         temperature=temperature,
#         min_dist=min_dist,
#         distinct_paths=distinct_paths,
#     )

#     # 去重检查（允许浮点误差，用近似判重）
#     def has_duplicates(X: Tensor, tol: float = 1e-9) -> bool:
#         for i in range(X.shape[0]):
#             for j in range(i + 1, X.shape[0]):
#                 if torch.allclose(X[i], X[j], atol=tol, rtol=0):
#                     return True
#         return False

#     if has_duplicates(Xq):
#         raise RuntimeError(f"{sampler_cls.__name__}: 批量 ask 返回了重复候选，请检查去重逻辑。")

#     Yq = toy_objective(Xq)
#     sampler.tell(Xq, Yq, refit=True)
#     report(f"After batch ask/tell (q={batch})")

#     # 最终再要一个点，快速 sanity
#     Xf = sampler.ask(
#         n=1,
#         num_paths=num_paths_single,
#         cand_size=cand_size,
#         temperature=temperature,
#         min_dist=min_dist,
#         distinct_paths=distinct_paths,
#     )
#     Yf = toy_objective(Xf)
#     sampler.tell(Xf, Yf, refit=True)
#     report("Final")

# def main():
#     # 这里导入你实现的三个类（如果在其他文件，请替换导入方式）
#     # from your_file import WeightedQEI, WeightedQLogEI, WeightedTS
#     # 逐个跑
#     for cls in [WeightedQEI, WeightedQLogEI, WeightedTS]:
#         run_one_sampler(cls, d=3, n_init=12, iters=3, batch=3)
# ======================== 测试先验：点偏好 & 值偏好 =========================
class PointBumpPrior:
    """
    在已知最优点 x* 附近为路径加权：对每条路径在候选上的“最接近 x* 的得分”加总。
    compute_norm_probs(...) -> (1,P,1)，和 = P
    """
    def __init__(self, x_star_norm: Tensor, sigma: float = 0.06, prior_floor: float = 1e-6):
        self.x_star = x_star_norm.detach()
        self.sigma2 = float(sigma) ** 2
        self.floor = float(prior_floor)

    @torch.no_grad()
    def compute_norm_probs(self, paths, raw_samples=2**12, **kwargs) -> Tensor:
        # 取一批 Sobol 候选，评估每条路径；用距离 x* 的 RBF 作为“位置得分”
        d = self.x_star.numel()
        eng = torch.quasirandom.SobolEngine(d, scramble=True, seed=777)
        C = eng.draw(raw_samples).to(dtype=self.x_star.dtype, device=self.x_star.device)  # [M,d]
        Y = paths(C.unsqueeze(-3)).squeeze(-1).squeeze(1)  # [P,M]
        # 位置分数（与路径值无关，仅位置偏好）：对每条路径，对所有候选求 sum(位置权)
        # 也可以把位置权 * relu(Y - y_best) 混合，这里先纯位置以示范先验效果
        diff2 = ((C - self.x_star) ** 2).sum(dim=-1)  # [M]
        w_pos_m = torch.exp(- diff2 / (2 * self.sigma2))  # [M]
        w_raw = w_pos_m.sum().expand(Y.shape[0])  # [P] ——纯位置先验 => 每条路径同权；为了演示，我们也可乘以路径自己的峰值
        # 更尖锐（可选）：若需要让更“能在 x* 一带给出高值”的路径更重权
        #w_raw = (torch.softmax(Y, dim=-1) * w_pos_m).sum(dim=-1)  # [P]

        w_raw = w_raw.clamp_min(self.floor)
        w = w_raw / (w_raw.sum() + 1e-12)  # 归一为 1
        P = w.numel()
        return (P * w).view(1, P, 1)

class ValuePeakPrior:
    """
    偏好“单条路径在候选上的最大值”接近 y*_std 的路径。
    兼容 BoTorch 0.15.1：不访问 MatheronPath 内部属性，完全通过前向推断 P 与输出形状。
    """
    def __init__(
        self,
        y_star_std: float,
        *,
        d: int,
        dtype: torch.dtype,
        device: torch.device,
        beta: float = 0.3,
        prior_floor: float = 1e-6,
        raw_samples: int = 2**12,  # 用于近似路径最大值的候选数
        sobol_seed: int = 778,
    ):
        self.y_star_std = float(y_star_std)
        self.beta2 = float(beta) ** 2
        self.floor = float(prior_floor)
        self.raw_samples = int(raw_samples)
        self.d = int(d)
        self.dtype = dtype
        self.device = device
        self.sobol_seed = int(sobol_seed)

    @torch.no_grad()
    def compute_norm_probs(self, paths, **kwargs) -> torch.Tensor:
        # 1) 用少量 probe 推断 P
        X_probe = torch.rand(3, self.d, dtype=self.dtype, device=self.device)  # [3,d]
        Y_probe = paths(X_probe.unsqueeze(-3)).squeeze(-1).squeeze(1)          # [P,3]
        P = Y_probe.shape[0]

        # 2) 用 Sobol 候选近似每条路径的最大值
        eng = torch.quasirandom.SobolEngine(self.d, scramble=True, seed=self.sobol_seed)
        C = eng.draw(self.raw_samples).to(dtype=self.dtype, device=self.device)    # [M,d]
        Y = paths(C.unsqueeze(-3)).squeeze(-1).squeeze(1)                          # [P,M]
        y_max = Y.max(dim=-1).values                                               # [P]

        # 3) 权重：高斯核逼近到 y*_std
        w_raw = torch.exp(- (y_max - self.y_star_std) ** 2 / (2 * self.beta2))     # [P]
        w_raw = w_raw.clamp_min(self.floor)
        w = w_raw / (w_raw.sum() + 1e-12)                                          # sum=1
        return (P * w).view(1, P, 1)                                               # 和 = P
# ========================= 仅修改 run_one_sampler：加入三种“偏好”模式 =========================
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

def run_one_sampler(
    sampler_cls: Type,
    d: int = 3,
    n_init: int = 12,
    iters: int = 3,
    batch: int = 3,
    *,
    num_paths_single: int = 512,
    num_paths_batch: int = 1024,
    cand_size: int = 8192,
    temperature: float = 1.0,
    min_dist: float | None = None,
    distinct_paths: bool = True,
    seed: int = 2025,
    sobol_seed: int = 42,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.double,
    # 偏好模式
    pref_mode: str = "none",  # "none" | "point" | "value" | "region"
    x_star_norm: Tensor | None = None,
    point_sigma: float = 0.06,
    value_beta: float = 0.35,
    region_box_radius: float = 0.08,
    region_delta: float = 0.6,
    region_smooth: float = 0.06,
):
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- 分块评估的最小间距贪心 -------
    def greedy_select_with_min_dist(acq_fn, C: Tensor, n: int, r_min: float | None, *, chunk_size: int = 256):
        """
        acq_fn: 接受 X_chunk: [K,1,d] -> 分数 [K,1] 或 [K]
        分块评估避免一次性把所有候选喂进模型，降低内存峰值。
        """
        M = C.size(0)
        device = C.device
        mask = torch.ones(M, dtype=torch.bool, device=device)
        selected_idx = []

        for _ in range(n):
            active = torch.arange(M, device=device)[mask]
            if active.numel() == 0:
                break

            # 分块计算每个候选的分数
            best_val = None
            best_j_rel = None
            for start in range(0, active.numel(), chunk_size):
                end = min(start + chunk_size, active.numel())
                idx_chunk = active[start:end]
                vals = acq_fn(C[idx_chunk].unsqueeze(1)).view(-1)  # [K]
                j_rel_chunk = int(vals.argmax().item())
                val_chunk = vals[j_rel_chunk]
                if (best_val is None) or (val_chunk > best_val):
                    best_val = val_chunk
                    best_j_rel = start + j_rel_chunk

            j = active[best_j_rel].item()
            selected_idx.append(j)

            # 最小间距屏蔽
            if r_min is None:
                mask[j] = False
            else:
                dists = torch.cdist(C[active], C[j:j+1]).squeeze(-1)  # [M_active]
                close = dists < float(r_min)
                mask[active[close]] = False

        return C[torch.tensor(selected_idx, device=device)]


    # ---------------- 边界与初始设计 ----------------
    bounds = torch.stack(
        [torch.zeros(d, dtype=dtype, device=device),
         torch.ones(d, dtype=dtype, device=device)], dim=0
    )
    eng = torch.quasirandom.SobolEngine(d, scramble=True, seed=sobol_seed)
    X0 = eng.draw(n_init).to(dtype=dtype, device=device)
    Y0 = toy_objective(X0)

    if x_star_norm is None:
        x_star_norm = torch.full((d,), 0.75, dtype=dtype, device=device)
    y_star = toy_objective(x_star_norm.view(1, -1)).item()

    # 实例化采样器
    sampler = sampler_cls(bounds=bounds, X_init=X0, Y_init=Y0, dtype=dtype, device=device)

    # 安装偏好（point/value 仍走你的路径加权；region 改走 MC qEI）
    if pref_mode == "point":
        sampler.set_location_prior(PointBumpPrior(x_star_norm=x_star_norm, sigma=point_sigma))
    elif pref_mode == "value":
        assert sampler._y_mean is not None and sampler._y_std is not None
        y_star_std = (torch.tensor(y_star, dtype=dtype, device=device) - sampler._y_mean) / sampler._y_std.clamp_min(1e-12)
        sampler.set_location_prior(
            ValuePeakPrior(
                y_star_std=float(y_star_std.item()),
                d=d, dtype=dtype, device=device,
                beta=value_beta, raw_samples=2**12
            )
        )

    print(
        f"\n==== {sampler_cls.__name__} | pref={pref_mode} | d={d}, init={n_init}, iters={iters} ====\n"
        f"[cfg] P(single)={num_paths_single}, P(batch)={num_paths_batch}, "
        f"M={cand_size}, temp={temperature}, min_dist={min_dist}, distinct_paths={distinct_paths}\n"
        f"[x*]={x_star_norm.cpu().numpy().round(4).tolist()}, y*≈{y_star:.4f}"
    )
    def report(tag: str):
        ybest = sampler.train_Y.max().item()
        print(f"{tag} | current best (raw) = {ybest:.4f} | n_obs = {sampler.train_X.shape[0]}")

    report("Start")

    # ---------------- 单点阶段 ----------------
    if pref_mode == "region":
        lb = (x_star_norm - region_box_radius).clamp(0.0, 1.0)
        ub = (x_star_norm + region_box_radius).clamp(0.0, 1.0)
        pp = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds, grid_size=256,  # ↓ 比 1024 更省内存
            smooth=0.06, dtype=dtype, device=device
        )
        pp.set_box_region(lb, ub)
        pp.fit_lambda_by_delta(base_model=sampler.model, delta=region_delta, observation_noise=False)
        # --- 数值护栏 #1：夹紧 λ（避免过大/过小导致数值不稳）---
        lam_max = 10.0   # 你可以调 5~10 之间
        pp.set_lambda(float(torch.clamp(torch.tensor(pp._lam, dtype=dtype, device=device), -lam_max, lam_max)))
        pp.prepare_cache(base_model=sampler.model)
        with torch.no_grad():
            G_in = (_box_mask_norm(pp.Xg, pp._region_lb, pp._region_ub)).sum().item()
            x_star_b = x_star_norm.view(1, -1).to(dtype=dtype, device=device)
            mu0 = sampler.model.posterior(x_star_b).mean.item()
            mu1 = TiltedModel(sampler.model, pp).posterior(x_star_b).mean.item()
            print(f"[tilt dbg] points_in_box={int(G_in)}, Δμ(x*)={mu1 - mu0:.4f}")
        effective = TiltedModel(sampler.model, pp).eval()
        y_best_std = sampler._current_best_std().item()

        acq_single = qLogExpectedImprovement(  # ← 替 qExpectedImprovement
            model=effective,
            best_f=y_best_std,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_paths_single]))
        )

        C = _make_sobol_grid_norm(d, cand_size, dtype, device)
        X1 = greedy_select_with_min_dist(acq_single, C, n=1, r_min=min_dist, chunk_size=256)
    else:
        X1 = sampler.ask(
            n=1, num_paths=num_paths_single,
            cand_size=cand_size, temperature=temperature, min_dist=min_dist, distinct_paths=distinct_paths,
        )

    Y1 = toy_objective(X1)
    sampler.tell(X1, Y1, refit=True)
    report("After 1-point ask/tell")

    # ---------------- 批量阶段 ----------------
    if pref_mode == "region":
        lb = (x_star_norm - region_box_radius).clamp(0.0, 1.0)
        ub = (x_star_norm + region_box_radius).clamp(0.0, 1.0)
        pp = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds, grid_size=512,  # 同样用 512
            smooth=region_smooth, dtype=dtype, device=device
        )
        pp.set_box_region(lb, ub)
        pp.fit_lambda_by_delta(base_model=sampler.model, delta=region_delta, observation_noise=False)
        lam_max = 10.0   # 你可以调 5~10 之间
        pp.set_lambda(float(torch.clamp(torch.tensor(pp._lam, dtype=dtype, device=device), -lam_max, lam_max)))
        pp.prepare_cache(base_model=sampler.model)
        with torch.no_grad():
            G_in = (_box_mask_norm(pp.Xg, pp._region_lb, pp._region_ub)).sum().item()
            x_star_b = x_star_norm.view(1, -1).to(dtype=dtype, device=device)
            mu0 = sampler.model.posterior(x_star_b).mean.item()
            mu1 = TiltedModel(sampler.model, pp).posterior(x_star_b).mean.item()
            print(f"[tilt dbg] points_in_box={int(G_in)}, Δμ(x*)={mu1 - mu0:.4f}")
        effective = TiltedModel(sampler.model, pp).eval()
        y_best_std = sampler._current_best_std().item()

        acq_batch = qLogExpectedImprovement(
            model=effective,
            best_f=y_best_std,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_paths_batch]))
        )

        C = _make_sobol_grid_norm(d, cand_size, dtype, device)
        Xq = greedy_select_with_min_dist(acq_batch, C, n=batch, r_min=min_dist, chunk_size=128)  # 批量时块可再小点

    else:
        Xq = sampler.ask(
            n=batch, num_paths=num_paths_batch,
            cand_size=cand_size, temperature=temperature, min_dist=min_dist, distinct_paths=distinct_paths,
        )

    # 批量去重检查
    def has_duplicates(X: Tensor, tol: float = 1e-9) -> bool:
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                if torch.allclose(X[i], X[j], atol=tol, rtol=0):
                    return True
        return False
    if has_duplicates(Xq):
        raise RuntimeError(f"{sampler_cls.__name__} ({pref_mode}): 批量 ask 返回了重复候选。")

    Yq = toy_objective(Xq)
    sampler.tell(Xq, Yq, refit=True)
    report(f"After batch ask/tell (q={batch})")

    # ---------------- 收尾单点 ----------------
    if pref_mode == "region":
        lb = (x_star_norm - region_box_radius).clamp(0.0, 1.0)
        ub = (x_star_norm + region_box_radius).clamp(0.0, 1.0)
        pp = LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=bounds, grid_size=512,
            smooth=region_smooth, dtype=dtype, device=device
        )
        pp.set_box_region(lb, ub)
        pp.fit_lambda_by_delta(base_model=sampler.model, delta=region_delta, observation_noise=False)
        lam_max = 10.0   # 你可以调 5~10 之间
        pp.set_lambda(float(torch.clamp(torch.tensor(pp._lam, dtype=dtype, device=device), -lam_max, lam_max)))
        pp.prepare_cache(base_model=sampler.model)
        with torch.no_grad():
            G_in = (_box_mask_norm(pp.Xg, pp._region_lb, pp._region_ub)).sum().item()
            x_star_b = x_star_norm.view(1, -1).to(dtype=dtype, device=device)
            mu0 = sampler.model.posterior(x_star_b).mean.item()
            mu1 = TiltedModel(sampler.model, pp).posterior(x_star_b).mean.item()
            print(f"[tilt dbg] points_in_box={int(G_in)}, Δμ(x*)={mu1 - mu0:.4f}")
        effective = TiltedModel(sampler.model, pp).eval()
        y_best_std = sampler._current_best_std().item()

        acq_single = qLogExpectedImprovement(
            model=effective,
            best_f=y_best_std,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_paths_single]))
        )

        C = _make_sobol_grid_norm(d, cand_size, dtype, device)
        Xf = greedy_select_with_min_dist(acq_single, C, n=1, r_min=min_dist, chunk_size=256)
    else:
        Xf = sampler.ask(
            n=1, num_paths=num_paths_single,
            cand_size=cand_size, temperature=temperature, min_dist=min_dist, distinct_paths=distinct_paths,
        )

    Yf = toy_objective(Xf)
    sampler.tell(Xf, Yf, refit=True)
    report("Final")
def main():
    for cls in [WeightedQEI, WeightedQLogEI, WeightedTS]:
        run_one_sampler(cls, pref_mode="point", point_sigma=0.06)
        run_one_sampler(cls, pref_mode="value", value_beta=0.35)
        run_one_sampler(cls, pref_mode="region", region_box_radius=0.08, region_delta=1.5)
if __name__ == "__main__":
    main()
