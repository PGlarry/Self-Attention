"""
s4_figures.py — Visualization for all three modules
Usage:
    python s4_figures.py

Reads CSV files from results/spectral/ and results/approx/ and results/probe/,
produces figures into figures/.

Figures generated:
    fig1_erank_heatmap.png      : layer × head effective rank heatmap (per model)
    fig2_spectral_decay.png     : singular value energy decay curves (representative layers)
    fig3_approx_boxplot.png     : approximation error distribution by method × layer group
    fig4_probe_scatter.png      : KL divergence scatter by (layer, head) per model
    fig5_probe_group_boxplot.png: selected-head probe KL by target group
    fig6_sparse_retention_curve.png: sparse_topk retention vs approximation error
"""
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR     = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "qwen3_8b":    "#2E86AB",
    "gemma3_4b":   "#A23B72",
    "ministral8b": "#F18F01",
}
MODEL_LABELS = {
    "qwen3_8b":    "Qwen3-8B",
    "gemma3_4b":   "Gemma3-4B-IT",
    "ministral8b": "Ministral-8B",
}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 220,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


def load_spectral() -> dict[str, pd.DataFrame]:
    dfs = {}
    for fpath in sorted((RESULTS_DIR / "spectral").glob("spectral_*.csv")):
        tag = fpath.stem.replace("spectral_", "")
        dfs[tag] = pd.read_csv(fpath, encoding="utf-8-sig")
    return dfs


def load_approx() -> dict[str, pd.DataFrame]:
    dfs = {}
    for fpath in sorted((RESULTS_DIR / "approx").glob("approx_*.csv")):
        tag = fpath.stem.replace("approx_", "")
        dfs[tag] = pd.read_csv(fpath, encoding="utf-8-sig")
    return dfs


def load_probe() -> dict[str, pd.DataFrame]:
    dfs = {}
    for fpath in sorted((RESULTS_DIR / "probe").glob("probe_*.csv")):
        if fpath.name == "probe_targets.csv":
            continue
        # filename: probe_{model_tag}_{task_id}.csv
        stem = fpath.stem
        tag = stem[len("probe_"):] if stem.startswith("probe_") else stem
        if "_" in tag:
            tag = tag.rsplit("_", 1)[0]
        dfs[tag] = pd.read_csv(fpath, encoding="utf-8-sig")
    return dfs


# ── Fig 1: erank heatmap (layer × head, mean over tasks) ─────────────────────

def fig1_erank_heatmap(spectral: dict[str, pd.DataFrame]):
    tags = list(spectral.keys())
    if not tags:
        print("  [skip] no spectral data")
        return

    n_models = len(tags)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        df = spectral[tag]
        pivot = df.groupby(["layer", "head"])["erank"].mean().reset_index()
        n_layers = pivot["layer"].max() + 1
        n_heads  = pivot["head"].max() + 1
        mat = np.full((n_layers, n_heads), np.nan)
        for _, row in pivot.iterrows():
            mat[int(row["layer"]), int(row["head"])] = row["erank"]

        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       cmap="viridis", origin="upper")
        ax.set_xlabel("Head index", fontsize=9)
        ax.set_ylabel("Layer index", fontsize=9)
        ax.set_title(MODEL_LABELS.get(tag, tag), fontsize=10, fontweight="bold",
                     color=MODEL_COLORS.get(tag, "black"))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    out = FIG_DIR / "fig1_erank_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── Fig 2: diag_conc vs layer (locality profile) ─────────────────────────────

def fig2_locality_profile(spectral: dict[str, pd.DataFrame]):
    if not spectral:
        print("  [skip] no spectral data")
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    for tag, df in spectral.items():
        mean_per_layer = df.groupby("layer")["diag_conc"].mean()
        x = mean_per_layer.index.values
        y = mean_per_layer.values
        color = MODEL_COLORS.get(tag, "gray")
        ax.plot(x, y, color=color, linewidth=1.8,
                label=MODEL_LABELS.get(tag, tag))

    ax.set_xlabel("Layer index", fontsize=9)
    ax.set_ylabel("Mean diagonal concentration", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "fig2_locality_profile.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── Fig 3: approximation error boxplot ───────────────────────────────────────

def fig3_approx_boxplot(approx: dict[str, pd.DataFrame]):
    if not approx:
        print("  [skip] no approx data")
        return

    # Use first available model for the figure (or loop all)
    for tag, df in approx.items():
        methods = df["method"].unique()
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 5),
                                 sharey=False)
        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            sub = df[df["method"] == method]
            params = sorted(sub["param"].unique())
            data   = [sub[sub["param"] == p]["frob_rel"].dropna().values for p in params]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=1.5))
            color = MODEL_COLORS.get(tag, "#888888")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(params) + 1))
            ax.set_xticklabels([str(p) for p in params], fontsize=8)
            ax.set_xlabel(f"param ({method})", fontsize=8)
            ax.set_ylabel("Frobenius rel. error" if ax == axes[0] else "", fontsize=8)
            ax.set_title(method, fontsize=9, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)

        out = FIG_DIR / f"fig3_approx_boxplot_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {out}")


# ── Fig 4: causal probe scatter ──────────────────────────────────────────────

def fig4_probe_scatter(probe: dict[str, pd.DataFrame]):
    if not probe:
        print("  [skip] no probe data")
        return

    for tag, df in probe.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        color = MODEL_COLORS.get(tag, "gray")

        n_layers = df["layer"].max() + 1
        sc = ax.scatter(df["layer"], df["kl_div"],
                        c=df["kl_div"], cmap="YlOrRd",
                        s=12, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, label="KL divergence").ax.tick_params(labelsize=7)

        ax.set_xlabel("Layer index", fontsize=9)
        ax.set_ylabel("KL(base || intervened)", fontsize=9)
        ax.set_title(MODEL_LABELS.get(tag, tag),
                     fontsize=10, fontweight="bold", color=color)
        ax.spines[["top", "right"]].set_visible(False)

        out = FIG_DIR / f"fig4_probe_scatter_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {out}")


# ── Fig 5: selected-head probe group comparison ──────────────────────────────

def fig5_probe_group_boxplot(probe: dict[str, pd.DataFrame]):
    if not probe:
        print("  [skip] no probe data")
        return

    frames = []
    for tag, df in probe.items():
        if "target_group" not in df.columns:
            continue
        sub = df[df["target_group"].fillna("") != ""].copy()
        if sub.empty:
            continue
        sub["model_tag"] = tag
        frames.append(sub)

    if not frames:
        print("  [skip] no selected-head probe groups")
        return

    all_probe = pd.concat(frames, ignore_index=True)
    groups = ["high_complex_global", "local_band", "low_rank_diffuse"]
    tags = list(probe.keys())

    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4), sharey=False)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        df = all_probe[all_probe["model_tag"] == tag]
        data = [df[df["target_group"] == group]["kl_div"].dropna().values for group in groups]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.4),
                        flierprops=dict(marker="o", markerfacecolor="white",
                                        markeredgecolor="#333333", markersize=3,
                                        alpha=0.45))
        color = MODEL_COLORS.get(tag, "#888888")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        rng = np.random.default_rng(7)
        for xpos, values in enumerate(data, start=1):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(np.full(len(values), xpos) + jitter, values,
                       s=12, color="#202020", alpha=0.35, linewidths=0,
                       zorder=3)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(["high\ncomplex", "local\nband", "low-rank\ndiffuse"], fontsize=8)
        ax.set_ylabel("KL(base || intervened)", fontsize=9)
        ax.set_title(MODEL_LABELS.get(tag, tag), fontsize=10, fontweight="bold",
                     color=color)
        ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "fig5_probe_group_boxplot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── Fig 6: sparse retention curve ────────────────────────────────────────────

def fig6_sparse_retention_curve(approx: dict[str, pd.DataFrame]):
    if not approx:
        print("  [skip] no approx data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for tag, df in approx.items():
        sub = df[df["method"] == "sparse_topk"].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby("param")
            .agg(frob_mean=("frob_rel", "mean"), kl_mean=("kl_mean", "mean"))
            .reset_index()
            .sort_values("param")
        )
        color = MODEL_COLORS.get(tag, "gray")
        label = MODEL_LABELS.get(tag, tag)
        axes[0].plot(grouped["param"], grouped["frob_mean"], marker="o",
                     linewidth=1.8, color=color, label=label)
        axes[1].plot(grouped["param"], grouped["kl_mean"], marker="o",
                     linewidth=1.8, color=color, label=label)

    axes[0].set_title("Frobenius Error", fontsize=10)
    axes[0].set_xlabel("Retained entries per row (%)", fontsize=9)
    axes[0].set_ylabel("Mean relative error", fontsize=9)
    axes[1].set_title("Row-wise KL", fontsize=10)
    axes[1].set_xlabel("Retained entries per row (%)", fontsize=9)
    axes[1].set_ylabel("Mean KL", fontsize=9)

    for ax in axes:
        ax.set_xticks([5, 10, 20, 40])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

    out = FIG_DIR / "fig6_sparse_retention_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def main():
    print("Loading results ...")
    spectral = load_spectral()
    approx   = load_approx()
    probe    = load_probe()

    print(f"  Spectral files: {list(spectral.keys())}")
    print(f"  Approx files : {list(approx.keys())}")
    print(f"  Probe files  : {list(probe.keys())}")

    print("\nGenerating figures ...")
    fig1_erank_heatmap(spectral)
    fig2_locality_profile(spectral)
    fig3_approx_boxplot(approx)
    fig4_probe_scatter(probe)
    fig5_probe_group_boxplot(probe)
    fig6_sparse_retention_curve(approx)

    print("\n✅ All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
