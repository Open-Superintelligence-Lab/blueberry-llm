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


def plot_all(csv_path: Path, out_dir: Path) -> None:
    _setup(out_dir)
    df = pd.read_csv(csv_path)
    plot_A1_perplexity_vs_experts(df, out_dir)
    plot_A2_learning_curves(df, out_dir)
    plot_A4_throughput(df, out_dir)
    plot_A6_router_entropy(df, out_dir)
    plot_B1_params_regime(df, out_dir)
    plot_C1_ppl_vs_flops(df, out_dir)


def main() -> None:
    p = argparse.ArgumentParser("Generate scaling plots from CSV")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="experiments/plots")
    args = p.parse_args()
    plot_all(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()

