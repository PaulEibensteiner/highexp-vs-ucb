from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import drd2_maxfinding_benchmark as bench


OUT_DIR = Path("sweep_outputs/round2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bigger queried budget than the initial fast sweep, but still tractable.
EXP_BASE: dict[str, Any] = {
    "pool_size": 1800,
    "seed_size": 64,
    "rounds": 18,
    "batch_k": 40,
    "train_epochs": 18,
    "train_batch_size": 8192,
    "pred_batch_size": 8192,
    "hidden_dim": 384,
    "depth": 5,
    "prior_mean": 0.92,
    "tau": 0.08,
    "dpp_candidates": 256,
    "shortlist_size": 5000,
    "use_amp": True,
    "use_tf32": True,
    "cache_graphs_on_device": True,
    "auto_tune_cuda_batch_sizes": False,
    "log_level": "WARNING",
}

ROBUST_SEEDS = [101, 123, 321, 777, 2024, 42, 314, 2718, 9001, 555]
REFINE_SEEDS = [101, 123, 321, 777, 2024]


def clone_cfg(**updates: Any) -> bench.BenchmarkConfig:
    cfg = bench.BenchmarkConfig(**EXP_BASE)
    for k, v in updates.items():
        setattr(cfg, k, v)
    return cfg


def get_method(results: list[bench.MethodResult], method_name: str) -> bench.MethodResult:
    for r in results:
        if r.method == method_name:
            return r
    raise ValueError(f"Method not found: {method_name}")


def run_benchmark_silent(cfg: bench.BenchmarkConfig, methods: list[str]) -> dict[str, Any]:
    # Suppress verbose round-by-round output while running large sweeps.
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        out = bench.run_benchmark(cfg, methods=methods)
    return out


def build_random_baseline(seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = clone_cfg(seed=seed)
        out = run_benchmark_silent(cfg, methods=["random"])
        rr = get_method(out["results"], "random")
        rows.append(
            {
                "seed": seed,
                "pool_max": float(np.max(out["scores"])),
                "rand_best_final": float(rr.best_so_far[-1]),
                "rand_regret_final": float(rr.simple_regret[-1]),
                "rand_q95": rr.threshold_q95,
                "rand_q98": rr.threshold_q98,
                "queries_final": int(rr.queries[-1]),
            }
        )
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.to_csv(OUT_DIR / "random_baseline.csv", index=False)
    df.to_json(OUT_DIR / "random_baseline.json", orient="records", indent=2)
    return df


def summarize_trials(trial_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    tmp = trial_df.copy()
    tmp["hp_q95_pen"] = tmp["hp_q95"].fillna(10**9)
    tmp["hp_q98_pen"] = tmp["hp_q98"].fillna(10**9)
    tmp["rand_q95_pen"] = tmp["rand_q95"].fillna(10**9)
    tmp["rand_q98_pen"] = tmp["rand_q98"].fillna(10**9)

    summary = (
        tmp.groupby(group_cols, as_index=False)
        .agg(
            seeds=("seed", "count"),
            hp_best_final_mean=("hp_best_final", "mean"),
            rand_best_final_mean=("rand_best_final", "mean"),
            gain_vs_random_mean=("gain_vs_random", "mean"),
            gain_vs_random_std=("gain_vs_random", "std"),
            beat_rate=("beats_random", "mean"),
            hp_regret_mean=("hp_regret_final", "mean"),
            rand_regret_mean=("rand_regret_final", "mean"),
            regret_delta_mean=("regret_delta", "mean"),
            hp_q95_hit_rate=("hp_q95", lambda s: float(s.notna().mean())),
            rand_q95_hit_rate=("rand_q95", lambda s: float(s.notna().mean())),
            hp_q98_hit_rate=("hp_q98", lambda s: float(s.notna().mean())),
            rand_q98_hit_rate=("rand_q98", lambda s: float(s.notna().mean())),
            hp_q95_mean_pen=("hp_q95_pen", "mean"),
            rand_q95_mean_pen=("rand_q95_pen", "mean"),
            hp_q98_mean_pen=("hp_q98_pen", "mean"),
            rand_q98_mean_pen=("rand_q98_pen", "mean"),
        )
        .sort_values(["gain_vs_random_mean", "beat_rate", "hp_regret_mean"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return summary


def run_config_sweep(
    name: str,
    seeds: list[int],
    grid: list[dict[str, Any]],
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_by_seed = {int(r.seed): r for r in baseline.itertuples(index=False)}
    rows: list[dict[str, Any]] = []

    for i, hp in enumerate(grid, start=1):
        print(f"[{name}] combo {i}/{len(grid)}: {hp}")
        for seed in seeds:
            cfg = clone_cfg(seed=seed, **hp)
            out = run_benchmark_silent(cfg, methods=["high_prior_dpp"])
            hr = get_method(out["results"], "high_prior_dpp")
            rb = baseline_by_seed[seed]
            rows.append(
                {
                    "seed": seed,
                    **hp,
                    "pool_max": float(rb.pool_max),
                    "hp_best_final": float(hr.best_so_far[-1]),
                    "rand_best_final": float(rb.rand_best_final),
                    "hp_regret_final": float(hr.simple_regret[-1]),
                    "rand_regret_final": float(rb.rand_regret_final),
                    "hp_q95": hr.threshold_q95,
                    "rand_q95": rb.rand_q95,
                    "hp_q98": hr.threshold_q98,
                    "rand_q98": rb.rand_q98,
                    "gain_vs_random": float(hr.best_so_far[-1] - rb.rand_best_final),
                    "regret_delta": float(hr.simple_regret[-1] - rb.rand_regret_final),
                    "beats_random": bool(hr.best_so_far[-1] > rb.rand_best_final),
                }
            )

    trials = pd.DataFrame(rows)
    summary = summarize_trials(trials, [k for k in grid[0].keys()])

    trials.to_csv(OUT_DIR / f"{name}_trials.csv", index=False)
    trials.to_json(OUT_DIR / f"{name}_trials.json", orient="records", indent=2)
    summary.to_csv(OUT_DIR / f"{name}_summary.csv", index=False)
    summary.to_json(OUT_DIR / f"{name}_summary.json", orient="records", indent=2)

    return trials, summary


def top_row(summary: pd.DataFrame) -> dict[str, Any]:
    row = summary.iloc[0]
    return {k: row[k] for k in summary.columns}


def main() -> None:
    print("Building random baseline...")
    baseline = build_random_baseline(ROBUST_SEEDS)

    print("Running robust validation sweep (10 seeds, top 2 prior configs)...")
    robust_grid = [
        {"lr": 5e-4, "weight_decay": 1e-4, "tau": 0.08, "dpp_candidates": 256},
        {"lr": 1e-3, "weight_decay": 1e-4, "tau": 0.08, "dpp_candidates": 256},
    ]
    robust_trials, robust_summary = run_config_sweep(
        name="robust_validation",
        seeds=ROBUST_SEEDS,
        grid=robust_grid,
        baseline=baseline,
    )

    print("Running narrow optimizer sweep (5 seeds)...")
    opt_grid = [
        {"lr": lr, "weight_decay": wd, "tau": 0.08, "dpp_candidates": 256}
        for lr in [7e-4, 1e-3, 1.5e-3]
        for wd in [3e-5, 1e-4, 3e-4]
    ]
    opt_trials, opt_summary = run_config_sweep(
        name="optimizer_refine",
        seeds=REFINE_SEEDS,
        grid=opt_grid,
        baseline=baseline[baseline["seed"].isin(REFINE_SEEDS)].copy(),
    )

    best_opt = top_row(opt_summary)
    best_lr = float(best_opt["lr"])
    best_wd = float(best_opt["weight_decay"])

    print("Running acquisition sweep (5 seeds)...")
    acq_grid = [
        {"lr": best_lr, "weight_decay": best_wd, "tau": tau, "dpp_candidates": dpp}
        for tau in [0.06, 0.08, 0.10]
        for dpp in [256, 384]
    ]
    acq_trials, acq_summary = run_config_sweep(
        name="acquisition_refine",
        seeds=REFINE_SEEDS,
        grid=acq_grid,
        baseline=baseline[baseline["seed"].isin(REFINE_SEEDS)].copy(),
    )

    best_acq = top_row(acq_summary)
    best_tau = float(best_acq["tau"])
    best_dpp = int(best_acq["dpp_candidates"])

    print("Running capacity sweep (5 seeds)...")
    cap_grid = [
        {
            "lr": best_lr,
            "weight_decay": best_wd,
            "tau": best_tau,
            "dpp_candidates": best_dpp,
            "hidden_dim": h,
            "depth": d,
        }
        for h in [384, 512]
        for d in [5, 6]
    ]
    cap_trials, cap_summary = run_config_sweep(
        name="capacity_refine",
        seeds=REFINE_SEEDS,
        grid=cap_grid,
        baseline=baseline[baseline["seed"].isin(REFINE_SEEDS)].copy(),
    )

    analysis = {
        "base_config": asdict(clone_cfg()),
        "robust_validation_top": top_row(robust_summary),
        "optimizer_refine_top": top_row(opt_summary),
        "acquisition_refine_top": top_row(acq_summary),
        "capacity_refine_top": top_row(cap_summary),
        "overall": {
            "robust_validation_overall_gain_mean": float(robust_trials["gain_vs_random"].mean()),
            "robust_validation_overall_beat_rate": float((robust_trials["gain_vs_random"] > 0).mean()),
            "optimizer_overall_gain_mean": float(opt_trials["gain_vs_random"].mean()),
            "acquisition_overall_gain_mean": float(acq_trials["gain_vs_random"].mean()),
            "capacity_overall_gain_mean": float(cap_trials["gain_vs_random"].mean()),
        },
    }

    with (OUT_DIR / "analysis_round2.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    with (OUT_DIR / "analysis_round2.txt").open("w", encoding="utf-8") as f:
        f.write("Round2 sweep analysis\n")
        f.write(f"Robust top: {analysis['robust_validation_top']}\n")
        f.write(f"Optimizer top: {analysis['optimizer_refine_top']}\n")
        f.write(f"Acquisition top: {analysis['acquisition_refine_top']}\n")
        f.write(f"Capacity top: {analysis['capacity_refine_top']}\n")
        f.write(f"Overall: {analysis['overall']}\n")

    print("Saved round2 sweep outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
