import os
import json
import time
import random
import re
import requests
import numpy as np
import torch  
import csv  # NEW
from torch.quasirandom import SobolEngine  
from prior_monte_carlo import WeightedQLogEI as DefaultSampler
from decide import decide_preference_cola_from_expert, decide_preference_tilt_from_expert
from copy import deepcopy
from boo import propose_points_from_plan
from typing import Any, Sequence, Iterable, List, Dict, Optional, Literal
from fun.toy_fun import toy_results, make_bo_normalizer
from datetime import datetime
from prompt import (
    make_pseudo_user_prompt_toy,
    PSEUDO_DATASET_SYSTEM_PROMPT,
    make_user_prompt_lnp3,
    make_user_prompt_crossed_barrel,
    make_user_prompt_fecr,
    make_user_prompt_sandwich,
    make_user_prompt_toy_bo,
    build_next_user_prompt,
    _parse_assistant,  
)
from prompt import TOY_SYSTEM_PROMPT as SYSTEM_PROMPT
PLAN_POLICY = os.getenv("PLAN_POLICY", "tilt").lower()  
DTYPE = torch.float32
N_INIT = int(os.getenv("N_INIT", "2"))  
RUN_BASELINE = bool(int(os.getenv("RUN_BASELINE", "1")))
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
TOY_CASE = os.getenv("TOY_CASE", "ackley    ").strip().lower()
if TOY_CASE not in {"rastrigin", "ackley", "griewank", "levy"}:
    raise ValueError(f"Invalid TOY_CASE={TOY_CASE}. Choose from rastrigin/ackley/griewank/levy")
RESULT_CSV = os.getenv(
    "RESULT_CSV",
    f"./results/toy_run_{PLAN_POLICY}_{int(RUN_BASELINE)}_{timestamp}.csv"
)


BASE_URL   = os.getenv("BASE_URL", "http://localhost:8000")
ENDPOINT   = "/api/v1/chat/completions"
API_KEY    = os.getenv("API_KEY", "")
DEVICE = torch.device("cpu")         

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))  
TEMP       = float(os.getenv("LLM_TEMP", "0.2"))
DO_SAMPLE  = bool(int(os.getenv("DO_SAMPLE", "0")))  

BATCH_Q    = int(os.getenv("BATCH_Q", "3"))


PRINT_LIMIT = int(os.getenv("PRINT_LIMIT", "3000"))  
LOG_FILE    = os.getenv("LOG_FILE", "chat_io_log.txt") 


FINAL_BLOCK_RE = re.compile(
    r"(?is)"
    r"(?:^\s{0,3}(?:#+\s*)?\[?\s*final\s+answer\s*\]?\s*:?\s*$)"
    r".*?"
    r"(\[point\s*,\s*\[.*?\]\s*,\s*[01](?:\.\d+)?\]|\[region\s*,\s*\[\[.*?\]\s*,\s*\[.*?\]\]\s*,\s*[01](?:\.\d+)?\])",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
BAREWORD_IN_LIST_RE = re.compile(
    r'(?P<prefix>(?:\[|,)\s*)'            
    r'(?P<tok>(?!point\b)(?!region\b)'    
    r'(?!true\b)(?!false\b)(?!null\b)'    
    r'[A-Za-z_][A-Za-z0-9_\-]*)'          
    r'(?P<suffix>\s*(?:,|\]))',         
    flags=re.IGNORECASE
)

def _normalize_expert_for_bo(expert: Dict[str, Any], *, func_name: str, d: int) -> Dict[str, Any]:

    norm = make_bo_normalizer(func_name, d)
    e = deepcopy(expert)


    for key in ("point", "x", "x_star"):
        if key in e and e[key] is not None:
            e[key] = norm.normalize_point(_to_list_floats(e[key]))


    if "region_box" in e and e["region_box"] is not None:
        lb, ub = e["region_box"]
        lb_n, ub_n = norm.normalize_region(_to_list_floats(lb), _to_list_floats(ub))
        e["region_box"] = (lb_n, ub_n)

    if (e.get("region_box") is None) and ("region_center" in e and "region_radius" in e and
                                          e["region_center"] is not None and e["region_radius"] is not None):
        c = _to_list_floats(e["region_center"])
        r = float(e["region_radius"])
        lb = [ci - r for ci in c]
        ub = [ci + r for ci in c]
        lb_n, ub_n = norm.normalize_region(lb, ub)
        e["region_box"] = (lb_n, ub_n)
        e.pop("region_center", None)
        e.pop("region_radius", None)

    return e

def sanitize_bracket_block(text: str) -> str:

    m = FINAL_BLOCK_RE.search(text)
    if not m:
        return text
    block = m.group(1)

    def repl(mm: re.Match) -> str:
        tok = mm.group("tok")
        return f'{mm.group("prefix")}"{tok}"{mm.group("suffix")}'

    fixed_block = BAREWORD_IN_LIST_RE.sub(repl, block)

    start, end = m.span(1)
    return text[:start] + fixed_block + text[end:]
def slice_first_final_answer(text: str) -> str:
    m = FINAL_BLOCK_RE.search(text)
    return text if not m else text[:m.end()]


def call_chat(system_prompt: str, user_prompt: str) -> str:

    url = BASE_URL.rstrip("/") + ENDPOINT
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "intern-s1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "do_sample": DO_SAMPLE,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _append_io_log(case: str, round_id: int, user_prompt: str, assistant_raw: str, cut: str, parsed: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CASE={case} ROUND={round_id}\n")
            f.write("- USER PROMPT (sent) -\n")
            f.write(user_prompt + "\n")
            f.write("- ASSISTANT RAW (received) -\n")
            f.write(assistant_raw + "\n")
            f.write("- ASSISTANT CUT (first Final Answer slice) -\n")
            f.write(cut + "\n")
            f.write("- PARSED -\n")
            f.write(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
    except Exception as e:
        print(f"[warn] fail to write log: {e}")

def _to_float_safe(x: Any):
    try:
        return float(str(x))
    except:
        return x 

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))

def _clip01(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except:
        return v

def pick_executed_points(parsed: dict, q: int = 1, jitter: float = 0.02) -> List[Sequence[Any]]:

    mode = parsed.get("mode")
    points: List[Sequence[Any]] = []

    if mode == "point":
        base = [_to_float_safe(v) for v in parsed["point"]]
        points.append(base)

        for i in range(1, max(1, q)):
            pert = []
            for v in base:
                if _is_num(v):
                    dv = random.uniform(-jitter, jitter)
                    pert.append(_clip01(v + dv))
                else:
                    pert.append(v)
            points.append(pert)

    elif mode == "region":
        lb, ub = parsed["lb"], parsed["ub"]

        center = []
        for a, b in zip(lb, ub):
            fa, fb = _to_float_safe(a), _to_float_safe(b)
            if _is_num(fa) and _is_num(fb):
                center.append(0.5 * (fa + fb))
            else:

                center.append(a)
        points.append(center)

        for i in range(1, max(1, q)):
            samp = []
            for a, b in zip(lb, ub):
                fa, fb = _to_float_safe(a), _to_float_safe(b)
                if _is_num(fa) and _is_num(fb):
                    lo, hi = (fa, fb) if fa <= fb else (fb, fa)
                    samp.append(random.uniform(lo, hi))
                else:
                    samp.append(a)
            points.append(samp)

    else:

        points = []

    return points


def fake_results(case: str, x: Sequence[Any] | Sequence[Sequence[Any]], seed: int):

    def _one(mu_case: str, i_seed: int):
        random.seed(i_seed)
        jitter = lambda mu, s: round(random.gauss(mu, s), 3)
        if case == "lnp3":
            return {
                "drug_loading": jitter(70, 3),
                "encap_efficiency": jitter(85, 2),
                "particle_diameter_nm": max(80.0, jitter(160, 15)),
            }
        if case == "crossed":
            return {"toughness": max(0.0, jitter(8.0, 0.6))}
        if case == "fecr":
            return {
                "VE_%": max(0.0, min(100.0, jitter(88, 3))),
                "CE_%": max(0.0, min(100.0, jitter(92, 2))),
                "Decay_rate": max(0.0, jitter(0.04, 0.01)),
            }
        if case == "sandwich":
            return {
                "Score": max(0.0, min(110.0, jitter(80, 5))),
                "kcal": max(200.0, jitter(600, 40)),
            }
        if case == "toy":
            return {"f": max(0.0, jitter(10, 2))}
        return {"score": jitter(0.5, 0.1)}


    is_batch = (len(x) > 0 and isinstance(x[0], (list, tuple, dict))) if isinstance(x, (list, tuple)) else False
    if is_batch:
        out = []
        for i, _ in enumerate(x):
            out.append(_one(case, seed * 1000 + i))  
        return out
    else:
        return _one(case, seed)

def _to_list_floats(x) -> list[float]:
    if x is None: return []
    if isinstance(x, (list, tuple)): return [float(v) for v in x]
    try:
        import torch
        if hasattr(x, "detach"):
            return [float(v) for v in x.detach().cpu().flatten().tolist()]
    except Exception:
        pass
    return [float(x)]

def build_expert_input_from_parsed(parsed: Dict[str, Any], *, func_name: str, d: int) -> list:

    mode = parsed.get("mode")
    conf = float(parsed.get("confidence", 0.5))
    norm = make_bo_normalizer(func_name, d)

    if mode == "point" and "point" in parsed:
        x = [float(v) for v in parsed["point"]]
        z = norm.normalize_point(x)
        return ["point", z, conf]

    if mode == "region" and "lb" in parsed and "ub" in parsed:
        lb = [float(v) for v in parsed["lb"]]
        ub = [float(v) for v in parsed["ub"]]
        lbz, ubz = norm.normalize_region(lb, ub)
        return ["region", [lbz, ubz], conf]


    return ["none"]

def first_user_prompt(case: str):
    if case == "lnp3":
        return make_user_prompt_lnp3(), {"d": 5}
    if case == "crossed":
        return make_user_prompt_crossed_barrel(), {"d": 4}
    if case == "fecr":
        return make_user_prompt_fecr(), {"d": 3}
    if case == "sandwich":
        return make_user_prompt_sandwich(), {"d": 20}
    if case == "toy":
        return make_user_prompt_toy_bo(TOY_CASE, d=6), {"d": 6, "func_name": TOY_CASE}
    raise ValueError(case)
def eval_toy_denorm(func_name: str, d: int, z, *, seed: int = 0, noise_std: float = 0.0):

    norm = make_bo_normalizer(func_name, d)

    is_batch = isinstance(z, (list, tuple)) and len(z) > 0 and isinstance(z[0], (list, tuple))
    if is_batch:
        x_list = [norm.denormalize_point(p) for p in z]
        return toy_results(func_name=func_name, x=x_list, noise_std=noise_std, seed=seed, clip_to_domain=True)
    else:
        x = norm.denormalize_point(z)
        return toy_results(func_name=func_name, x=x, noise_std=noise_std, seed=seed, clip_to_domain=True)

CSV_HEADER = ["round", "track", "best_f_so_far", "best_x_so_far"]

def _csv_init(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

def _csv_log(path: str, round_id: int, track: str, best_f: float, best_x: list):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([round_id, track, best_f, json.dumps(best_x)])





def _extract_json_blob(text: str) -> str | None:

    s = text.strip()


    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)


    try:
        json.loads(s)
        return s
    except Exception:
        pass


    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def _parse_pseudo_json(text: str, expected_k: int | None = None, d: int | None = None):
    blob = _extract_json_blob(text)
    if not blob:

        raise ValueError("No JSON blob found in pseudo dataset response.")

    data = json.loads(blob)

    kd = int(data.get("k", 0))
    d_json = int(data.get("d", 0))
    pts = data.get("points", [])
    if not isinstance(pts, list):
        raise ValueError("'points' must be a list")

    if expected_k is not None and kd != expected_k:
        print(f"[warn] pseudo JSON k={kd} != expected {expected_k}; will truncate/pad.")

    clean = []
    for it in pts:
        try:
            x = it.get("x", [])
            est_f = float(it.get("est_f", 0.0))
            conf = float(it.get("confidence", 0.5))
        except Exception:
            continue
        if isinstance(x, list):
            clean.append({"x": x, "est_f": est_f, "confidence": max(0.0, min(1.0, conf))})

    return {"d_json": d_json, "k_json": kd, "points": clean}

def _ensure_k_points(points: list, k: int, d: int, bounds: list[tuple[float,float]]):
    pts = points[:k]
    if len(pts) < k:
        ests = [p["est_f"] for p in pts] or [0.0]
        med = sorted(ests)[len(ests)//2]
        while len(pts) < k:
            x = [random.uniform(lo, hi) for (lo,hi) in bounds]
            pts.append({"x": x, "est_f": float(med), "confidence": 0.5})

    for p in pts:
        x = p["x"]
        if len(x) != d:

            x = (x + [0.0]*d)[:d]
            p["x"] = x
    return pts

def get_pseudo_dataset(func_name: str, d: int, bounds, k: int,
                       base_url: str, api_key: str, endpoint: str,
                       model_name: str,
                       temp: float, max_tokens: int) -> tuple[list[dict], str]:
    user_prompt = make_pseudo_user_prompt_toy(func_name, d, bounds=bounds, k=k)
    url = base_url.rstrip("/") + endpoint
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": PSEUDO_DATASET_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
            {"response_format": {"type": "json_object"}},
        ],
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    raw = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    print("RAW assistant output (trunc):")
    print(raw)
    parsed = _parse_pseudo_json(raw, expected_k=k, d=d)

    default_bounds = {
        "rastrigin": (-5.12, 5.12),
        "ackley": (-5.0, 5.0),
        "griewank": (-600.0, 600.0),
        "levy": (-10.0, 10.0),
    }
    lo, hi = default_bounds.get(func_name, (-10.0, 10.0))
    bounds_full = [(lo, hi)] * d

    pts = _ensure_k_points(parsed["points"], k, d, bounds_full)
    return pts, raw 



def run_case(case: str, rounds: int = 5, sleep_s: float = 0.4):

    SOBOL_SEED = int(os.getenv("SOBOL_SEED", "42"))
    TORCH_SEED = int(os.getenv("TORCH_SEED", "42"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    
    torch.manual_seed(TORCH_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print(f"\n========== CASE: {case} ==========")
    print(f"Random seeds - Sobol: {SOBOL_SEED}, Torch: {TORCH_SEED}, Random: {RANDOM_SEED}")
    
    user_prompt, meta = first_user_prompt(case)
    d = meta["d"]

    if case != "toy":
        print("This run only tests 'toy'. Skipping:", case)
        return

    func_name = meta.get("func_name", TOY_CASE)
    normalizer = make_bo_normalizer(func_name, d)


    _csv_init(RESULT_CSV)


    sobol = SobolEngine(dimension=d, scramble=True, seed=SOBOL_SEED)
    Z0 = sobol.draw(max(1, N_INIT)).to(dtype=DTYPE, device=DEVICE)
    y0_list = eval_toy_denorm(func_name, d, [z.tolist() for z in Z0], seed=0)
    Y0 = torch.tensor([[float(r["f"])] for r in y0_list], dtype=DTYPE, device=DEVICE)


    X_hist = Z0.clone()
    y_hist = Y0.clone()

    if RUN_BASELINE:
        Xb_hist = Z0.clone()
        yb_hist = Y0.clone()

    def _best_of(X, y):
        vals = y.view(-1).tolist()
        idx = int(min(range(len(vals)), key=lambda i: vals[i]))
        return idx, float(vals[idx]), X[idx].tolist()

    _, best_f_llm, best_z_llm = _best_of(X_hist, y_hist)
    best_x_llm = normalizer.denormalize_point(best_z_llm)
    if RUN_BASELINE:
        _, best_f_base, best_z_base = _best_of(Xb_hist, yb_hist)
        best_x_base = normalizer.denormalize_point(best_z_base)

    _csv_log(RESULT_CSV, 0, "llm_bo",  best_f_llm,  best_x_llm)
    if RUN_BASELINE:
        _csv_log(RESULT_CSV, 0, "pure_bo", best_f_base, best_x_base)
    FORMAT_GUARD = (
        "\n\n[Formatting Guard]\n"
        "- WRITE 'Final Answer' FIRST, then a 1–2 sentence 'Thinking'.\n"
        "- 'Final Answer' must be the strict bracketed structure on one block.\n"
        "- Do not write anything else.\n"
    )

    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        up = user_prompt + FORMAT_GUARD


        raw = call_chat(SYSTEM_PROMPT, up)
        cut = slice_first_final_answer(raw)
        print("RAW assistant output (trunc):")
        print((cut if len(cut) <= PRINT_LIMIT else cut[:PRINT_LIMIT] + "\n... [truncated]"))
        cut_for_parse = sanitize_bracket_block(cut)
        parsed = _parse_assistant(cut_for_parse)
        mode = parsed.get("mode")
        conf = parsed.get("confidence")
        print(f"Parsed: mode={mode}, conf={conf}")
        _append_io_log(case, r, up, raw, cut, parsed)

        if not mode:
            print("!! No Final Answer parsed. Retry once with stricter guard.")
            stricter_guard = (
                "\n\n[Formatting Guard]\n"
                "- Write ONLY two blocks exactly as specified. If you wrote anything else, ignore it and rewrite now.\n"
                "- Thinking ≤ 80 tokens.\n"
            )
            raw2 = call_chat(SYSTEM_PROMPT, user_prompt + stricter_guard)
            cut2 = slice_first_final_answer(raw2)
            print("RAW assistant output (retry, trunc):")
            print((cut2 if len(cut2) <= PRINT_LIMIT else cut2[:PRINT_LIMIT] + "\n... [truncated]"))
            parsed2 = _parse_assistant(cut2)
            mode2 = parsed2.get("mode")
            conf2 = parsed2.get("confidence")
            print(f"Parsed(retry): mode={mode2}, conf={conf2}")
            _append_io_log(case, r, user_prompt + stricter_guard, raw2, cut2, parsed2)
            if mode2:
                parsed, mode, conf = parsed2, mode2, conf2
            else:
                print("[warn] LLM invalid twice; fallback to pure BO (plan=none).")
                parsed = {"mode": "none", "confidence": 0.0}

        expert_input = build_expert_input_from_parsed(parsed, func_name=func_name, d=d)
        try:
            if PLAN_POLICY == "tilt":
                plan = decide_preference_tilt_from_expert(expert_input, d=d, grid_size=512)
            else:
                plan = decide_preference_cola_from_expert(expert_input, d=d, grid_size=512)
        except Exception as e:
            print(f"[warn] decide_preference_* failed ({e}); fallback to plan=none.")
            plan = {"mode": "none", "confidence": float(conf or 0.0)}


        Z_new = propose_points_from_plan(
            sampler_cls=DefaultSampler,
            X=X_hist, y=-y_hist, plan=plan, q=max(1, BATCH_Q),
        )
        if Z_new.ndim == 1:
            Z_new = Z_new.unsqueeze(0)

        z_list = [z.tolist() for z in Z_new]
        y_meas_list = eval_toy_denorm(func_name, d, z_list, seed=r)
        y_new = torch.tensor([[float(r_["f"])] for r_ in y_meas_list], dtype=DTYPE, device=DEVICE)
        X_hist = torch.cat([X_hist, Z_new.to(dtype=DTYPE, device=DEVICE)], dim=0)
        y_hist = torch.cat([y_hist, y_new], dim=0)

        for zi, yi in zip(z_list, y_meas_list):
            fi = float(yi["f"])
            if fi < best_f_llm:
                best_f_llm, best_z_llm = fi, zi
                best_x_llm = normalizer.denormalize_point(best_z_llm)
        _csv_log(RESULT_CSV, r, "llm_bo", best_f_llm, best_x_llm)

        if RUN_BASELINE:
            Zb_new = propose_points_from_plan(
                sampler_cls=DefaultSampler,
                X=Xb_hist, y=-yb_hist, plan={"mode": "none"}, q=max(1, BATCH_Q),
            )
            if Zb_new.ndim == 1:
                Zb_new = Zb_new.unsqueeze(0)
            zb_list = [z.tolist() for z in Zb_new]
            yb_meas_list = eval_toy_denorm(func_name, d, zb_list, seed=10_000 + r)
            yb_new = torch.tensor([[float(r_["f"])] for r_ in yb_meas_list], dtype=DTYPE, device=DEVICE)
            Xb_hist = torch.cat([Xb_hist, Zb_new.to(dtype=DTYPE, device=DEVICE)], dim=0)
            yb_hist = torch.cat([yb_hist, yb_new], dim=0)

            for zi, yi in zip(zb_list, yb_meas_list):
                fi = float(yi["f"])
                if fi < best_f_base:
                    best_f_base, best_z_base = fi, zi
                    best_x_base = normalizer.denormalize_point(best_z_base)
            _csv_log(RESULT_CSV, r, "pure_bo", best_f_base, best_x_base)

        x_exec_list_real = [normalizer.denormalize_point(z) for z in z_list]
        user_prompt = build_next_user_prompt(
            make_user_prompt_fn=make_user_prompt_toy_bo,
            prev_user_prompt=user_prompt,
            assistant_text=cut,
            executed_point=(x_exec_list_real if len(x_exec_list_real) > 1 else x_exec_list_real[0]),
            results=([{"f": float(v.item())} for v in y_new] if len(x_exec_list_real) > 1 else {"f": float(y_new[0].item())}),
            key_observations=f"auto-note r{r}: BO results ingested (policy={PLAN_POLICY}).",
            this_round_focus="Balance exploration & exploitation.",
            last_reasoning_override=parsed.get("thinking"),
            include_old_history=True,
            history_keep_n=None,
            func_name=func_name,
            d=d,
        )

        print("\n[Next USER prompt preview (trunc)]")
        print((user_prompt if len(user_prompt) <= PRINT_LIMIT else user_prompt[:PRINT_LIMIT] + "\n... [truncated]"))
        time.sleep(sleep_s)


def main():
    for case in ["toy"]:
        run_case(case, rounds=30)

if __name__ == "__main__":
    main()