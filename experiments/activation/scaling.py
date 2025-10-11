import numpy as np
from torch.utils.data import Subset
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from llm import train_moe_model, MoEModelConfig

# Scaling law function (only data term + constant since model size N is fixed)
def scaling_law(D, a, b, c):
    """
    L(D) = a / (D^b) + c
    - a : coefficient for data scaling
    - b : scaling exponent (key metric!)
    - c : irreducible error (loss floor)
    """
    return a / (D ** b) + c


# -------------------------------
# Extended Benchmark
# -------------------------------
def benchmark_activation_scaling(act_name: str, dataset, tokenizer, dataset_fracs=[0.1, 0.3, 1.0], num_steps=200):
    """
    Run scaling-law benchmark for one activation across different dataset sizes.
    """
    results = []
    total_len = len(dataset)

    for frac in dataset_fracs:
        size = int(total_len * frac)
        indices = list(range(size))
        sub_dataset = Subset(dataset, indices)
        val_size = size // 10
        train_size = size - val_size

        train_dataset, val_dataset = random_split(sub_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Config (small, fixed model size)
        config = MoEModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=1024,
            num_experts=4,
            expert_top_k=2,
            num_documents=size,
            max_tokens=20000,
            max_seq_len=128,
            batch_size=16,
            max_steps=num_steps,
            eval_every=20,
        )
        config.activation = act_name
        config.vocab_size = len(tokenizer)

        # Train + collect results
        _, final_eval, history = train_moe_model(config, train_loader, val_loader)
        results.append({
            "activation": act_name,
            "frac": frac,
            "dataset_size": size,
            "val_loss": final_eval["val_loss"],
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "val_steps": history["val_steps"]
        })

    return results


# -------------------------------
# Fit scaling exponents
# -------------------------------
def fit_scaling_exponent(results):
    """
    Fit power-law curve: L(D) = a / D^b + c
    Returns exponent b and fit parameters per activation.
    """
    summary = {}

    grouped = {}
    for r in results:
        grouped.setdefault(r["activation"], []).append(r)

    for act, act_results in grouped.items():
        Ds = np.array([r["dataset_size"] for r in act_results], dtype=float)
        Ls = np.array([r["val_loss"] for r in act_results], dtype=float)

        # Fit curve
        popt, _ = curve_fit(scaling_law, Ds, Ls, maxfev=10000)
        a, b, c = popt
        summary[act] = {"a": a, "b": b, "c": c, "raw": (Ds, Ls)}

    return summary


# -------------------------------
# Plot scaling curves
# -------------------------------
def plot_scaling_curves(summary, save_path="scaling_laws.png"):
    plt.figure(figsize=(8,6))

    for act, params in summary.items():
        Ds, Ls = params["raw"]
        sorted_idx = np.argsort(Ds)
        Ds_sorted = Ds[sorted_idx]
        Ls_sorted = Ls[sorted_idx]

        # Fitted curve
        fit_D = np.logspace(np.log10(min(Ds)), np.log10(max(Ds)), 50)
        fit_L = scaling_law(fit_D, params["a"], params["b"], params["c"])

        plt.plot(Ds_sorted, Ls_sorted, "o", label=f"{act} (val)")
        plt.plot(fit_D, fit_L, "--", label=f"{act} fit (b={params['b']:.2f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset Size (log)")
    plt.ylabel("Validation Loss (log)")
    plt.title("Scaling Laws: Activation Functions")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📉 Saved scaling-law plot to {save_path}")
