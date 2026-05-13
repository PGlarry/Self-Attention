"""
s5_select_probe_targets.py — Select heads for structured causal probing.

Usage:
    python s5_select_probe_targets.py [heads_per_group]

Reads:
    results/spectral/spectral_{model_tag}.csv

Writes:
    results/probe/probe_targets.csv

Selection logic:
    For each model, aggregate spectral metrics over all prompts for each
    (layer, head). Select:
      - high_complex_global: top-quartile erank, then long distance and low diagonal concentration
      - local_band         : high diagonal concentration, short mean distance
      - low_rank_diffuse   : bottom-quartile erank, then long distance and low diagonal concentration

These groups operationalize the RQ3 contrast without assuming that low rank
necessarily means locality. In the current data, very low-erank heads are often
diffuse/long-distance rather than local, which is itself an interpretable class.
"""
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SPECTRAL_DIR = RESULTS_DIR / "spectral"
PROBE_DIR = RESULTS_DIR / "probe"
OUT_CSV = PROBE_DIR / "probe_targets.csv"

HEADS_PER_GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def select_for_model(path: Path) -> pd.DataFrame:
    model_tag = path.stem.replace("spectral_", "")
    df = pd.read_csv(path, encoding="utf-8-sig")

    agg = (
        df.groupby(["model_tag", "layer", "head"], as_index=False)
        .agg(
            erank=("erank", "mean"),
            stable_rank=("stable_rank", "mean"),
            energy_top1=("energy_top1", "mean"),
            energy_top5=("energy_top5", "mean"),
            diag_conc=("diag_conc", "mean"),
            mean_dist=("mean_dist", "mean"),
            seq_len=("seq_len", "mean"),
        )
        .sort_values(["layer", "head"])
    )

    agg["score_high_complex_global"] = (
        zscore(agg["erank"]) + zscore(agg["mean_dist"]) - zscore(agg["diag_conc"])
    )
    agg["score_local_band"] = zscore(agg["diag_conc"]) - zscore(agg["mean_dist"])
    agg["score_low_rank_diffuse"] = (
        -zscore(agg["erank"]) + zscore(agg["mean_dist"]) - zscore(agg["diag_conc"])
    )

    high_candidates = agg[agg["erank"] >= agg["erank"].quantile(0.75)].copy()
    high = (
        high_candidates.sort_values("score_high_complex_global", ascending=False)
        .head(HEADS_PER_GROUP)
        .copy()
    )
    high["target_group"] = "high_complex_global"
    used = set(zip(high["layer"], high["head"]))

    local_candidates = agg[~agg.apply(lambda r: (r["layer"], r["head"]) in used, axis=1)].copy()
    local = (
        local_candidates.sort_values("score_local_band", ascending=False)
        .head(HEADS_PER_GROUP)
        .copy()
    )
    local["target_group"] = "local_band"
    used.update(zip(local["layer"], local["head"]))

    low_candidates = agg[
        (agg["erank"] <= agg["erank"].quantile(0.25))
        & (~agg.apply(lambda r: (r["layer"], r["head"]) in used, axis=1))
    ].copy()
    low_rank = (
        low_candidates.sort_values("score_low_rank_diffuse", ascending=False)
        .head(HEADS_PER_GROUP)
        .copy()
    )
    low_rank["target_group"] = "low_rank_diffuse"

    out = pd.concat([high, local, low_rank], ignore_index=True)
    out["selection_rank"] = out.groupby("target_group").cumcount() + 1
    out["source_file"] = path.name
    out["model_tag"] = model_tag
    return out[
        [
            "model_tag",
            "target_group",
            "selection_rank",
            "layer",
            "head",
            "erank",
            "stable_rank",
            "energy_top1",
            "energy_top5",
            "diag_conc",
            "mean_dist",
            "score_high_complex_global",
            "score_local_band",
            "score_low_rank_diffuse",
            "seq_len",
            "source_file",
        ]
    ]


def main():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SPECTRAL_DIR.glob("spectral_*.csv"))
    if not files:
        raise SystemExit(f"No spectral CSV files found in {SPECTRAL_DIR}")

    selected = pd.concat([select_for_model(path) for path in files], ignore_index=True)
    selected.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Selected {len(selected)} probe targets -> {OUT_CSV}")
    for model_tag, sub in selected.groupby("model_tag"):
        print(f"\n[{model_tag}]")
        cols = ["target_group", "selection_rank", "layer", "head", "erank", "diag_conc", "mean_dist"]
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
