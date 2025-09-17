import matplotlib.pyplot as plt
import numpy as np
import math
# -------------------------------
# 1. Subplots: train vs val loss per activation
# -------------------------------
def plot_loss_subplots(results, save_path="loss_subplots.png"):
    grouped = {}
    for r in results:
        grouped.setdefault(r["activation"], []).append(r)

    n_acts = len(grouped)
    fig, axes = plt.subplots(n_acts, 1, figsize=(8, 3*n_acts), sharex=True)

    if n_acts == 1:
        axes = [axes]

    for ax, (act, act_results) in zip(axes, grouped.items()):
        for run in act_results:
            steps = list(range(1, len(run["train_losses"]) + 1))
            ax.plot(steps,[math.log(i) for i in run["train_losses"]], label=f"{act} train (frac={run['frac']})")
            if run["val_losses"]:
                ax.plot(run["val_steps"],[math.log(i) for i in run["val_losses"]], linestyle="--",
                        label=f"{act} val (frac={run['frac']})")
        ax.set_title(f"Activation: {act}")
        ax.set_ylabel("Loss")
        ax.legend()

    axes[-1].set_xlabel("Training Steps")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📉 Saved subplots to {save_path}")


# -------------------------------
# 2. Bar plot: compute efficiency
# -------------------------------
def plot_compute_bar(results, save_path="compute_bar.png"):
    acts = []
    times = []
    for r in results:
        if "time_per_step_sec" in r:
            acts.append(f"{r['activation']} (frac={r['frac']})")
            times.append(r["time_per_step_sec"])

    plt.figure(figsize=(8,5))
    plt.bar(acts, times)
    plt.ylabel("Time per Step (s)")
    plt.title("Compute Cost per Activation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Saved compute bar plot to {save_path}")


# -------------------------------
# 3. Scaling: val loss vs dataset size
# -------------------------------
def plot_loss_vs_dataset(results, save_path="scaling_dataset.png"):
    grouped = {}
    for r in results:
        grouped.setdefault(r["activation"], []).append(r)

    plt.figure(figsize=(8,6))
    for act, act_results in grouped.items():
        Ds = np.array([r["dataset_size"] for r in act_results], dtype=float)
        Ls = np.array([r["val_loss"] for r in act_results], dtype=float)

        order = np.argsort(Ds)
        Ds, Ls = Ds[order], Ls[order]

        plt.plot(Ds, Ls, "o-", label=f"{act}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset Size (log)")
    plt.ylabel("Validation Loss (log)")
    plt.title("Scaling: Validation Loss vs Dataset Size")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📉 Saved scaling plot to {save_path}")
