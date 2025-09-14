from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_A1_perplexity_vs_experts(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=df, x="num_experts", y="val_perplexity", hue="seed", marker="o", ax=ax, estimator=None)
    ax.set_title("Plot A1: Perplexity vs Experts (Fixed FLOPs)")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "A1_ppl_vs_experts.png")


def plot_A2_learning_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=df, x="tokens_seen", y="val_perplexity", hue="num_experts", marker="o", ax=ax)
    ax.set_title("Plot A2: Learning Curves @Fixed FLOPs")
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "A2_learning_curves.png")


def plot_A3_active_params_vs_ppl(df: pd.DataFrame, out_dir: Path) -> None:
    if "active_params_per_token" not in df:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.scatterplot(
        data=df,
        x="active_params_per_token",
        y="val_perplexity",
        hue="num_experts",
        ax=ax,
    )
    ax.set_title("Plot A3: PPL vs Active Params/Token")
    ax.set_xlabel("Active Params/Token (proxy)")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "A3_ppl_vs_active_params.png")


def plot_A4_throughput(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.barplot(data=df, x="num_experts", y="tokens_per_second", ax=ax)
    ax.set_title("Plot A4: Throughput vs Experts (Fixed FLOPs)")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Tokens/s")
    _save(fig, out_dir / "A4_throughput_vs_experts.png")


def plot_A6_router_entropy(df: pd.DataFrame, out_dir: Path) -> None:
    if "router_entropy_mean" not in df:
        return
    df2 = df.dropna(subset=["router_entropy_mean"]).copy()
    if df2.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=df2, x="num_experts", y="router_entropy_mean", marker="o", ax=ax)
    ax.set_title("Plot A6: Router Entropy vs Experts")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Mean Router Entropy")
    _save(fig, out_dir / "A6_router_entropy_vs_experts.png")


def plot_A5_expert_usage_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    if "router_usage_frac" not in df:
        return
    dff = df.dropna(subset=["router_usage_frac"]).copy()
    if dff.empty:
        return
    for e, g in dff.groupby("num_experts"):
        # Average usage vectors across rows
        vecs = g["router_usage_frac"].apply(lambda v: eval(v) if isinstance(v, str) else v)
        # Normalize each then average
        import numpy as np

        arr = np.stack([np.array(v, dtype=float) for v in vecs])
        arr = arr / arr.sum(axis=1, keepdims=True)
        mean = arr.mean(axis=0)

        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        sns.barplot(x=list(range(len(mean))), y=mean, ax=ax, color="#4C72B0")
        ax.set_title(f"Plot A5: Expert Usage — E={e}")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Fraction of tokens")
        _save(fig, out_dir / f"A5_expert_usage_E{e}.png")


def plot_B1_params_regime(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    dfp = df[df["regime"] == "params"].copy()
    if dfp.empty:
        return
    sns.lineplot(data=dfp, x="num_experts", y="val_perplexity", marker="o", ax=ax)
    ax.set_title("Plot B1: Perplexity vs Experts (Fixed Params)")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "B1_ppl_vs_experts_fixed_params.png")


def plot_C1_ppl_vs_flops(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.scatterplot(
        data=df,
        x="flops_per_token_forward",
        y="val_perplexity",
        hue="num_experts",
        style="regime",
        ax=ax,
    )
    ax.set_title("Plot C1: Perplexity vs FLOPs/Token")
    ax.set_xlabel("Fwd FLOPs/Token (proxy)")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "C1_ppl_vs_flops.png")


def plot_B2_ppl_vs_flops_params(df: pd.DataFrame, out_dir: Path) -> None:
    dfp = df[df["regime"] == "params"].copy()
    if dfp.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.scatterplot(data=dfp, x="flops_per_token_forward", y="val_perplexity", hue="num_experts", ax=ax)
    ax.set_title("Plot B2: PPL vs FLOPs/Token (Iso-Params)")
    ax.set_xlabel("Fwd FLOPs/Token (proxy)")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "B2_ppl_vs_flops_params.png")


def plot_B3_flops_vs_experts_params(df: pd.DataFrame, out_dir: Path) -> None:
    dfp = df[df["regime"] == "params"].copy()
    if dfp.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=dfp, x="num_experts", y="flops_per_token_forward", marker="o", ax=ax)
    ax.set_title("Plot B3: FLOPs/Token vs Experts (Fixed Params)")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Fwd FLOPs/Token (proxy)")
    _save(fig, out_dir / "B3_flops_vs_experts_params.png")


def plot_C2_ppl_vs_total_params(df: pd.DataFrame, out_dir: Path) -> None:
    if "total_expert_params" not in df:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.scatterplot(data=df, x="total_expert_params", y="val_perplexity", hue="num_experts", style="regime", ax=ax)
    ax.set_title("Plot C2: PPL vs Total Expert Params")
    ax.set_xlabel("Total expert params")
    ax.set_ylabel("Validation PPL")
    _save(fig, out_dir / "C2_ppl_vs_total_params.png")


def plot_C3_heatmap_ppl_E_dff(df: pd.DataFrame, out_dir: Path) -> None:
    # Mean across seeds
    d = df.groupby(["num_experts", "d_ff"], as_index=False)["val_perplexity"].mean()
    if d.empty:
        return
    pivot = d.pivot(index="num_experts", columns="d_ff", values="val_perplexity")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("Plot C3: Heatmap PPL over (E, d_ff_expert)")
    ax.set_xlabel("d_ff_expert")
    ax.set_ylabel("Experts E")
    _save(fig, out_dir / "C3_heatmap_ppl_E_dff.png")


def plot_D1_peak_mem_vs_experts(df: pd.DataFrame, out_dir: Path) -> None:
    if "peak_mem_bytes" not in df:
        return
    df2 = df.copy()
    df2["peak_mem_gb"] = df2["peak_mem_bytes"].astype(float) / (1024 ** 3)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=df2, x="num_experts", y="peak_mem_gb", marker="o", ax=ax)
    ax.set_title("Plot D1: Peak Memory vs Experts")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("Peak memory (GB)")
    _save(fig, out_dir / "D1_peak_mem_vs_experts.png")


def plot_D2_seconds_per_mtokens(df: pd.DataFrame, out_dir: Path) -> None:
    if "tokens_per_second" not in df:
        return
    df2 = df.copy()
    df2["sec_per_1M_tokens"] = 1e6 / df2["tokens_per_second"].replace(0, float("nan"))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=df2, x="num_experts", y="sec_per_1M_tokens", marker="o", ax=ax)
    ax.set_title("Plot D2: Seconds per 1M Tokens vs Experts")
    ax.set_xlabel("Experts E")
    ax.set_ylabel("sec / 1M tokens")
    _save(fig, out_dir / "D2_sec_per_1M_tokens_vs_experts.png")


def plot_all(csv_path: Path, out_dir: Path) -> None:
    _setup(out_dir)
    df = pd.read_csv(csv_path)
    plot_A1_perplexity_vs_experts(df, out_dir)
    plot_A2_learning_curves(df, out_dir)
    plot_A3_active_params_vs_ppl(df, out_dir)
    plot_A4_throughput(df, out_dir)
    plot_A5_expert_usage_histogram(df, out_dir)
    plot_A6_router_entropy(df, out_dir)
    plot_B1_params_regime(df, out_dir)
    plot_C1_ppl_vs_flops(df, out_dir)
    plot_B2_ppl_vs_flops_params(df, out_dir)
    plot_B3_flops_vs_experts_params(df, out_dir)
    plot_C2_ppl_vs_total_params(df, out_dir)
    plot_C3_heatmap_ppl_E_dff(df, out_dir)
    plot_D1_peak_mem_vs_experts(df, out_dir)
    plot_D2_seconds_per_mtokens(df, out_dir)


def main() -> None:
    p = argparse.ArgumentParser("Generate scaling plots from CSV")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="experiments/plots")
    args = p.parse_args()
    plot_all(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
