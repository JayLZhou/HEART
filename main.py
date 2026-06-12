
from Option.Config2 import Config
import argparse
import os
import random
import json
import copy
import numpy as np
import torch
from pathlib import Path
from shutil import copyfile
from collections import defaultdict
from Data.DataLoader import RAGDataset
from Common.Constants import PROJECT_ROOT
from Common.Utils import welcome_message
from tqdm import tqdm
from Common.Logger import logger
from Pipeline.FlowBuild import FlowBuilder
from Tuner.TunerFactory import get_tuner
from Tuner.BOTuner.lgbo_components.budget_aware import BudgetAwareAllocator, ClusterRoundStats
from Utils.Evaluation import Evaluator
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from optuna.distributions import BaseDistribution
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from Tuner.BOTuner.lgbo_components.search_space import NumericSearchSpaceAdapter, CategoricalSearchSpaceAdapter
from Tuner.BOTuner.lgbo_components.unified_surrogate import build_surrogate_layers, params_to_unit
from Tuner.BOTuner.OptunaTuner import wrap_params
from openai import OpenAI
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def _sanitize_proxy_env() -> None:
    # Some runtime deps rely on httpx which rejects socks5h:// in older builds.
    # Rewrite to socks5:// to keep proxy behavior while avoiding startup crashes.
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(key)
        if value and value.startswith("socks5h://"):
            os.environ[key] = "socks5://" + value[len("socks5h://") :]


_sanitize_proxy_env()
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
args = parser.parse_args()

opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
builder = FlowBuilder(config=opt)
dataset = RAGDataset(data_dir=os.path.join(opt.data_root, opt.dataset_name))
num_trials = opt.num_trials




def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_path = Path(args.opt)
    opt_name = opt_path.name
    basic_name = PROJECT_ROOT / "Option" / "Config2.yaml"
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(str(basic_name), os.path.join(config_dir, "Config2.yaml"))
    return result_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_secret(value: str | None) -> str | None:
    if not value:
        return value
    return os.path.expandvars(value)


def _build_clusters_from_dataset() -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for i in range(len(dataset)):
        q = dataset[i]
        cid = _cluster_key(q)
        q["__cluster_id"] = cid
        clusters[cid].append(q)
    return dict(clusters)


def _assign_kmeans_clusters_if_enabled() -> None:
    if not getattr(opt.tuner.optimization, "cluster_kmeans_enabled", False):
        return

    out_path = os.path.join(opt.working_dir, opt.exp_name, "Results", "kmeans_clustered_questions.jsonl")

    # Load from cache if already computed (avoids re-embedding 1000 queries)
    if os.path.exists(out_path):
        import pandas as pd
        cached = pd.read_json(out_path, orient="records", lines=True)
        if "cluster_id" in cached.columns and len(cached) == len(dataset.dataset):
            dataset.dataset["cluster_id"] = cached["cluster_id"].values
            logger.info(f"K-means clustering loaded from cache: {out_path}")
            return

    emb_cfg = opt.embedding
    api_key = _resolve_secret(getattr(emb_cfg, "api_key", None))
    base_url = getattr(emb_cfg, "base_url", None)
    model = getattr(emb_cfg, "model", None)
    dims = getattr(emb_cfg, "dimensions", None)
    if not api_key or not base_url or not model:
        raise RuntimeError("Embedding config is incomplete for K-means clustering.")

    logger.info("Running K-means clustering on question embeddings...")
    client = OpenAI(api_key=api_key, base_url=base_url)
    questions = dataset.dataset["question"].astype(str).tolist()
    batch_size = int(getattr(emb_cfg, "embed_batch_size", 64) or 64)

    embs: list[list[float]] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        kwargs = {"model": model, "input": batch}
        if dims:
            kwargs["dimensions"] = int(dims)
        rsp = client.embeddings.create(**kwargs)
        embs.extend([d.embedding for d in rsp.data])

    X = np.array(embs, dtype=np.float32)
    k = int(getattr(opt.tuner.optimization, "cluster_kmeans_k", 5))
    seed = int(getattr(opt.tuner.optimization, "cluster_kmeans_random_state", 42))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    dataset.dataset["cluster_id"] = labels.astype(int)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.dataset.to_json(out_path, orient="records", lines=True, force_ascii=False)
    logger.info(f"K-means clustering finished. Saved clustered questions to {out_path}")


def _default_flow_flat_config() -> dict:
    defaults = dict(opt.tuner.search_space._defaults())
    defaults.update(opt.tuner.search_space.reranker.defaults())
    available_llms = [llm.model for llm in getattr(opt, "llms", []) if getattr(llm, "model", None)]
    if available_llms:
        defaults["response_synthesizer_llm"] = available_llms[0]
        if "rag_query_decomposition_llm_name" in defaults:
            defaults["rag_query_decomposition_llm_name"] = available_llms[0]
    return defaults


def _evaluate_full_per_cluster(*, clusters: dict[str, list[dict]], flat_config_by_cluster: dict[str, dict], tag: str):
    def _eval_cluster(cid: str, qlist: list[dict], flow_cfg: dict) -> dict:
        flow = builder.build_flow(wrap_params(flow_cfg))
        correct = 0
        for i, q in enumerate(qlist, start=1):
            q_eval = copy.deepcopy(q)
            try:
                q_eval["output"] = flow.query(q_eval["question"])
                m = evaluator.evaluate_single(q_eval)
                correct += 1 if float(m["accuracy"]) >= 50 else 0
            except Exception as ex:
                logger.warning(f"[FullEval:{tag}] cluster={cid} query_idx={i} failed: {ex}")
            if i % 25 == 0:
                logger.info(f"[FullEval:{tag}] cluster={cid} progress {i}/{len(qlist)}")
        full_acc = 100.0 * correct / max(1, len(qlist))
        logger.info(f"[FullEval:{tag}] cluster={cid} acc={full_acc:.2f}% ({correct}/{len(qlist)})")
        return {
            "cluster_id": int(cid) if str(cid).isdigit() else cid,
            "tag": tag,
            "trial_id": None,
            "train_acc_sampled": None,
            "full_eval_n": len(qlist),
            "full_acc": full_acc,
        }

    rows = []
    items = sorted(clusters.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
    max_workers = max(1, min(5, len(items)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for cid, qlist in items:
            flow_cfg = flat_config_by_cluster.get(cid) or _default_flow_flat_config()
            futures.append(ex.submit(_eval_cluster, cid, qlist, flow_cfg))
        for fut in as_completed(futures):
            rows.append(fut.result())
    return sorted(rows, key=lambda r: r["cluster_id"])


def _extract_best_flow_by_cluster(tuner, cluster_ids: list[str]) -> dict[str, dict]:
    best_by_cluster: dict[str, dict] = {}
    all_trials = tuner.completed_trials()
    for cid in cluster_ids:
        local = [
            tr for tr in all_trials
            if str(((getattr(tr, "user_attrs", {}) or {}).get("query") or {}).get("__cluster_id")) == str(cid)
        ]
        if not local:
            local = all_trials
        if not local:
            continue
        best = max(local, key=lambda tr: float(tr.values[0] if tr.values else tr.value))
        flow_blob = (getattr(best, "user_attrs", {}) or {}).get("flow")
        if isinstance(flow_blob, str):
            flow_cfg = json.loads(flow_blob)
        else:
            flow_cfg = flow_blob
        best_by_cluster[str(cid)] = flow_cfg
    return best_by_cluster


def _plot_training_acc_curve(tuner, result_dir: str) -> None:
    clusters: dict[str, list[float]] = defaultdict(list)
    for tr in tuner.completed_trials():
        attrs = getattr(tr, "user_attrs", {}) or {}
        q = attrs.get("query") or {}
        cid = str(q.get("__cluster_id", q.get("cluster_id", "unknown")))
        acc = attrs.get("metric_accuracy")
        if acc is None:
            continue
        clusters[cid].append(float(acc))

    if not clusters:
        return

    plt.figure(figsize=(9, 5), dpi=150)
    for cid in sorted(clusters.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        ys = clusters[cid]
        xs = list(range(1, len(ys) + 1))
        plt.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.5, label=f"cluster {cid} (n={len(ys)})")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Per-cluster training step")
    plt.ylabel("Training accuracy")
    plt.title("Training Accuracy Curves by Cluster")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    out = os.path.join(result_dir, "training_acc_curve_by_cluster.png")
    plt.savefig(out)
    logger.info(f"Saved training acc curve: {out}")


def wrapper_tuning():
    logger.info("Starting RAG tuning: query level")
    if opt.tuner.optimization.sampler == "lgbo":
        return wrapper_tuning_budget_aware_lgbo()
    if opt.tuner.optimization.sampler in {"tpe", "hierarchical"}:
        return wrapper_tuning_tpe_flat()
    if opt.tuner.optimization.sampler == "gpbo":
        return wrapper_tuning_gpbo_flat()
    if opt.tuner.optimization.sampler == "llm_tpe":
        return wrapper_tuning_llm_tpe_flat()
    if opt.tuner.optimization.sampler == "llambo":
        return wrapper_tuning_llambo_flat()

    dataset_len = len(dataset)
    for _, idx in enumerate(range(dataset_len)):
        results = []
        query = dataset[idx]
        tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=query)
        for i in tqdm(range(num_trials), desc="Running trials"):
            logger.info(f"Running trial {i+1}/{num_trials}")
            try:
                result = tuner(query = query)
            except Exception as e:
                logger.error(f"Trial {i+1} failed with error: {str(e)}")
                raise
                continue
        results.append(result)
    return {"tuner": tuner, "clusters": _build_clusters_from_dataset()}

def _cluster_key(query: dict) -> str:
    for key in ("cluster_id", "cluster", "query_cluster"):
        if key in query and query[key] is not None:
            return str(query[key])
    return str(query["id"])


def _extract_region_center_and_size_from_trial(trial, dim: int) -> tuple[list[float], list[float]]:
    attrs = getattr(trial, "user_attrs", {}) or {}
    plan = attrs.get("lgbo_plan") or {}
    lower = plan.get("lower") or {}
    upper = plan.get("upper") or {}
    point = plan.get("point") or {}

    keys = sorted(set(lower.keys()) | set(upper.keys()) | set(point.keys()))
    if not keys:
        center = [0.5 for _ in range(dim)]
        half = [0.25 for _ in range(dim)]
        return center, half

    center = []
    half = []
    for k in keys[:dim]:
        if k in lower and k in upper:
            lo = float(lower[k]); hi = float(upper[k])
            center.append((lo + hi) / 2.0)
            half.append(abs(hi - lo) / 2.0)
        elif k in point:
            center.append(float(point[k]))
            half.append(0.02)
        else:
            center.append(0.5)
            half.append(0.25)
    while len(center) < dim:
        center.append(0.5); half.append(0.25)
    return center, half


def _extract_confidence_from_trial(trial) -> float:
    attrs = getattr(trial, "user_attrs", {}) or {}
    parsed = attrs.get("lgbo_preference_parsed") or {}
    plan = attrs.get("lgbo_plan") or {}
    c = parsed.get("confidence", plan.get("confidence", 0.5))
    try:
        return max(0.0, min(1.0, float(c)))
    except Exception:
        return 0.5


def wrapper_tuning_tpe_flat():
    """TPE baseline: no cluster structure, flat round-robin over all queries."""
    logger.info("Starting flat TPE baseline tuning (no clustering)")
    queries = [dataset[i] for i in range(len(dataset))]
    for q in queries:
        q["__cluster_id"] = "all"

    if not queries:
        return None

    first_query = queries[0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)

    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(queries))
    rr_ptr = 0

    for t in range(1, T + 1):
        logger.info(f"[TPE-Flat] Round {t}/{T}, budget={B}")
        context = {"cluster_id": "all", "allocated_budget": B, "round": t}
        selected = []
        for _ in range(B):
            selected.append(queries[rr_ptr % len(queries)])
            rr_ptr += 1
        try:
            tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
        except Exception as e:
            logger.error(f"[TPE-Flat] Round {t} failed: {str(e)}")
            raise

    return {"tuner": tuner, "clusters": {"all": queries}}


def wrapper_tuning_gpbo_flat():
    """Flat GP-BO baseline (no LLM guidance, no clustering) — inspired by Barker et al. 2025."""
    logger.info("Starting flat GPBO baseline tuning (GP surrogate, no LLM, no clustering)")
    queries = [dataset[i] for i in range(len(dataset))]
    for q in queries:
        q["__cluster_id"] = "all"
    first_query = queries[0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)
    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(queries))
    rr_ptr = 0
    for t in range(1, T + 1):
        logger.info(f"[GPBO-Flat] Round {t}/{T}, budget={B}")
        context = {"cluster_id": "all", "allocated_budget": B, "round": t}
        selected = []
        for _ in range(B):
            selected.append(queries[rr_ptr % len(queries)])
            rr_ptr += 1
        try:
            tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
        except Exception as e:
            logger.error(f"[GPBO-Flat] Round {t} failed: {str(e)}")
            raise
    return {"tuner": tuner, "clusters": _build_clusters_from_dataset()}


def wrapper_tuning_llm_tpe_flat():
    """Flat LLM-TPE baseline (LLAMBO-inspired: LLM as config generator, no clustering)."""
    logger.info("Starting flat LLM-TPE baseline tuning (LLAMBO-inspired, no clustering)")
    queries = [dataset[i] for i in range(len(dataset))]
    for q in queries:
        q["__cluster_id"] = "all"
    first_query = queries[0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)
    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(queries))
    rr_ptr = 0
    for t in range(1, T + 1):
        logger.info(f"[LLM-TPE-Flat] Round {t}/{T}, budget={B}")
        context = {"cluster_id": "all", "allocated_budget": B, "round": t}
        selected = []
        for _ in range(B):
            selected.append(queries[rr_ptr % len(queries)])
            rr_ptr += 1
        try:
            tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
        except Exception as e:
            logger.error(f"[LLM-TPE-Flat] Round {t} failed: {str(e)}")
            raise
    return {"tuner": tuner, "clusters": _build_clusters_from_dataset()}



def wrapper_tuning_llambo_flat():
    """Flat LLAMBO baseline: LLM as zero-shot surrogate (Ma et al., 2024)."""
    logger.info("Starting flat LLAMBO baseline tuning (LLM as surrogate, no clustering)")
    queries = [dataset[i] for i in range(len(dataset))]
    for q in queries:
        q["__cluster_id"] = "all"
    first_query = queries[0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)
    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(queries))
    rr_ptr = 0
    for t in range(1, T + 1):
        logger.info(f"[LLAMBO-Flat] Round {t}/{T}, budget={B}")
        context = {"cluster_id": "all", "allocated_budget": B, "round": t}
        selected = []
        for _ in range(B):
            selected.append(queries[rr_ptr % len(queries)])
            rr_ptr += 1
        try:
            tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
        except Exception as e:
            logger.error(f"[LLAMBO-Flat] Round {t} failed: {str(e)}")
            raise
    return {"tuner": tuner, "clusters": _build_clusters_from_dataset()}

def wrapper_tuning_budget_aware_tpe():
    """TPE baseline: cluster-round budget allocation with uniform distribution (no GP utility)."""
    logger.info("Starting budget-aware cluster-round TPE baseline tuning")
    queries = [dataset[i] for i in range(len(dataset))]
    clusters: dict[str, list[dict]] = {}
    for q in queries:
        cid = _cluster_key(q)
        q["__cluster_id"] = cid
        clusters.setdefault(cid, []).append(q)

    cluster_ids = sorted(clusters.keys())
    if not cluster_ids:
        return None

    first_query = clusters[cluster_ids[0]][0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)

    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(cluster_ids))
    K = len(cluster_ids)
    n_min = int(opt.tuner.optimization.budget_n_min)
    rr_ptr = {cid: 0 for cid in cluster_ids}

    for t in range(1, T + 1):
        logger.info(f"[TPE-Cluster] Round {t}/{T}")
        base = max(n_min, B // K)
        alloc: dict[str, int] = {cid: base for cid in cluster_ids}
        remainder = B - base * K
        for cid in cluster_ids[:max(0, remainder)]:
            alloc[cid] += 1
        logger.info(f"[TPE-Cluster] allocation={alloc}")

        for cid in cluster_ids:
            n_queries = int(alloc.get(cid, 0))
            if n_queries <= 0:
                continue
            context = {
                "cluster_id": cid,
                "allocated_budget": n_queries,
                "round": t,
            }
            qlist = clusters[cid]
            selected = []
            for _ in range(n_queries):
                selected.append(qlist[rr_ptr[cid] % len(qlist)])
                rr_ptr[cid] += 1
            try:
                tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
            except Exception as e:
                logger.error(f"[TPE-Cluster] Cluster {cid} round-trial failed: {str(e)}")
                raise

    return {"tuner": tuner, "clusters": clusters}


def wrapper_tuning_budget_aware_lgbo():
    logger.info("Starting budget-aware cluster-round LGBO tuning")
    queries = [dataset[i] for i in range(len(dataset))]
    clusters: dict[str, list[dict]] = {}
    for q in queries:
        cid = _cluster_key(q)
        q["__cluster_id"] = cid
        clusters.setdefault(cid, []).append(q)

    cluster_ids = sorted(clusters.keys())
    if not cluster_ids:
        return None

    first_query = clusters[cluster_ids[0]][0]
    tuner = get_tuner(config=opt, builder=builder, evaluator=evaluator, query=first_query)
    lgbo_sampler = tuner.get_sampler()
    full_search_space: dict[str, BaseDistribution] = lgbo_sampler.infer_relative_search_space(study=None, trial=None)
    numeric_adapter = NumericSearchSpaceAdapter()
    categorical_adapter = CategoricalSearchSpaceAdapter()
    numeric_specs = numeric_adapter.build_specs(numeric_adapter.filter_numeric_distributions(full_search_space))
    categorical_specs = categorical_adapter.build_specs(
        categorical_adapter.filter_categorical_distributions(full_search_space, exclude_names=set())
    )
    surrogate_specs, metas = build_surrogate_layers(numeric_specs, categorical_specs)
    allocator = BudgetAwareAllocator(
        n_min=max(1, int(opt.tuner.optimization.budget_n_min)),
        tau=float(opt.tuner.optimization.budget_tau),
        ema_alpha=float(opt.tuner.optimization.budget_ema_alpha),
        warm_start_synergy_threshold=float(opt.tuner.optimization.warm_start_synergy_threshold),
    )

    T = int(opt.tuner.optimization.budget_rounds or opt.num_trials)
    B = int(opt.tuner.optimization.budget_per_round or len(cluster_ids))
    # Strictly follow design doc: first round uses uniform allocation (tau -> inf behavior).
    cold_rounds = 1
    gamma = float(opt.tuner.optimization.budget_gamma)
    rr_ptr = {cid: 0 for cid in cluster_ids}

    for t in range(1, T + 1):
        logger.info(f"[BudgetAware-LGBO] Round {t}/{T}")
        completed = tuner.completed_trials()
        param_dim = len(getattr(completed[-1], "params", {}) or {}) if completed else 8

        cluster_stats: dict[str, ClusterRoundStats] = {}
        cluster_best: dict[str, dict | None] = {}
        for cid in cluster_ids:
            local_trials = [
                tr for tr in completed
                if str(((getattr(tr, "user_attrs", {}) or {}).get("query") or {}).get("__cluster_id")) == cid
            ]
            if local_trials:
                objs = [float(tr.values[0] if tr.values else tr.value) for tr in local_trials]
                last_t = local_trials[-1]
                conf = _extract_confidence_from_trial(last_t)
                center, half = _extract_region_center_and_size_from_trial(last_t, dim=max(param_dim, len(surrogate_specs)))
                region_eff = 0.0
                region_points: list[list[float]] = []
                var_mean = float(np.var(objs)) + 1e-6
                if len(local_trials) >= 2 and metas:
                    try:
                        tx = []
                        ty = []
                        for tr in local_trials:
                            unit = params_to_unit(getattr(tr, "params", {}) or {}, metas)
                            tx.append([unit[m.name] for m in metas])
                            ty.append(float(tr.values[0] if tr.values else tr.value))
                        X = torch.tensor(tx, dtype=torch.double)
                        Y = torch.tensor(ty, dtype=torch.double).unsqueeze(-1)
                        Y = (Y - Y.mean(0, keepdim=True)) / Y.std(0, unbiased=False, keepdim=True).clamp_min(1e-12)
                        model = SingleTaskGP(train_X=X, train_Y=Y)
                        mll = ExactMarginalLogLikelihood(model.likelihood, model)
                        fit_gpytorch_mll(mll)
                        model.eval()
                        grid = torch.quasirandom.SobolEngine(len(metas), scramble=True).draw(256).to(dtype=torch.double)
                        post = model.posterior(grid)
                        v = post.variance.squeeze(-1)
                        if len(center) == len(metas):
                            c = torch.tensor(center, dtype=torch.double)
                            h = torch.tensor(half, dtype=torch.double)
                            lb = (c - h).clamp(0.0, 1.0)
                            ub = (c + h).clamp(0.0, 1.0)
                            mask = ((grid >= lb) & (grid <= ub)).all(dim=-1)
                            if mask.any():
                                var_mean = float(v[mask].mean().item())
                                # region effective size: sqrt(a^T Σ_GG a), with uniform a.
                                cov = post.mvn.covariance_matrix
                                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                                Kgg = cov[idx][:, idx]
                                n_g = int(Kgg.shape[0])
                                a = torch.full((n_g, 1), 1.0 / max(n_g, 1), dtype=torch.double)
                                region_eff = float(torch.sqrt((a.t() @ Kgg @ a).squeeze().clamp_min(1e-12)).item())
                                # Keep a compact region point set for RKHS-synergy computation.
                                region_points = grid[idx].tolist()
                            else:
                                var_mean = float(v.mean().item())
                                region_eff = float(math.sqrt(sum(hh * hh for hh in half)))
                                region_points = grid[:32].tolist()
                        else:
                            var_mean = float(v.mean().item())
                            region_eff = float(math.sqrt(sum(hh * hh for hh in half)))
                            region_points = grid[:32].tolist()
                    except Exception:
                        region_eff = float(math.sqrt(sum(hh * hh for hh in half)))
                        region_points = [[float(x) for x in center]]
                else:
                    region_eff = float(math.sqrt(sum(hh * hh for hh in half)))
                    region_points = [[float(x) for x in center]]
                best_t = max(local_trials, key=lambda tr: float(tr.values[0] if tr.values else tr.value))
                cluster_best[cid] = dict(getattr(best_t, "params", {}) or {})
            else:
                var_mean = 1.0
                conf = 0.5
                center = [0.5] * param_dim
                half = [0.25] * param_dim
                region_eff = float(math.sqrt(sum(h * h for h in half)))
                region_points = [center]
                cluster_best[cid] = None

            cluster_stats[cid] = ClusterRoundStats(
                cluster_id=cid,
                confidence=conf,
                region_center=center,
                region_half_width=half,
                region_points=region_points,
                posterior_var_mean=var_mean,
                region_effective_size=region_eff,
            )

        cluster_stats = allocator.estimate_utilities(cluster_stats)
        utilities = {cid: st.utility for cid, st in cluster_stats.items()}
        alloc = allocator.allocate(
            B=B,
            cluster_ids=cluster_ids,
            utilities=utilities,
            cold_start=(t <= cold_rounds),
        )
        logger.info(f"[BudgetAware-LGBO] allocation={alloc}")

        for cid in cluster_ids:
            n_queries = int(alloc.get(cid, 0))
            if n_queries <= 0:
                continue
            source_cluster, synergy = allocator.best_transfer_source(
                target_cluster_id=cid,
                cluster_stats=cluster_stats,
                cluster_best=cluster_best,
            )
            warm_start_candidate = cluster_best.get(source_cluster) if source_cluster else None
            context = {
                "cluster_id": cid,
                "utility": float(cluster_stats[cid].utility),
                "gamma": gamma,
                "warm_start_candidate": warm_start_candidate,
                "warm_start_source_cluster": source_cluster,
                "warm_start_synergy": float(synergy),
                "allocated_budget": n_queries,
                "round": t,
            }

            qlist = clusters[cid]
            selected = []
            for _ in range(n_queries):
                selected.append(qlist[rr_ptr[cid] % len(qlist)])
                rr_ptr[cid] += 1
            try:
                tuner.run_cluster_trial(cluster_queries=selected, cluster_context=context)
            except Exception as e:
                logger.error(f"Cluster {cid} round-trial failed with error: {str(e)}")
                raise
    return {"tuner": tuner, "clusters": clusters}


if __name__ == "__main__":
    welcome_message()
    seed_everything(42)
    result_dir = check_dirs(opt)

    # Offline indexing
    corpus = dataset.get_corpus()
    builder.build_indexing(corpus)
    evaluator = Evaluator(eval_path=os.path.join(opt.working_dir, opt.exp_name, "Results", "results.json"), dataset_name=opt.dataset_name)
    _assign_kmeans_clusters_if_enabled()

    full_eval_rows = []
    if getattr(opt.tuner.optimization, "run_full_eval_before_after", False):
        pre_clusters = _build_clusters_from_dataset()
        init_cfg = _default_flow_flat_config()
        pre_cfg_by_cluster = {cid: init_cfg for cid in pre_clusters.keys()}
        full_eval_rows.extend(
            _evaluate_full_per_cluster(
                clusters=pre_clusters,
                flat_config_by_cluster=pre_cfg_by_cluster,
                tag="pretrain_initial",
            )
        )

    # Online RAG tuning
    tune_out = wrapper_tuning()
    if tune_out and tune_out.get("tuner"):
        try:
            _plot_training_acc_curve(tune_out["tuner"], result_dir)
        except Exception as _plot_e:
            logger.warning(f"Plot failed (non-fatal): {_plot_e}")
    if getattr(opt.tuner.optimization, "run_full_eval_before_after", False) and tune_out:
        post_clusters = tune_out.get("clusters") or _build_clusters_from_dataset()
        best_cfg_by_cluster = _extract_best_flow_by_cluster(tune_out["tuner"], list(post_clusters.keys()))
        full_eval_rows.extend(
            _evaluate_full_per_cluster(
                clusters=post_clusters,
                flat_config_by_cluster=best_cfg_by_cluster,
                tag="posttrain_best_per_cluster",
            )
        )
        full_eval_path = os.path.join(result_dir, "full_eval_before_after_by_cluster.json")
        with open(full_eval_path, "w", encoding="utf-8") as f:
            json.dump(full_eval_rows, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved before/after full eval report: {full_eval_path}")
