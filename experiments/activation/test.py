import time
import matplotlib.pyplot as plt
from enum import Enum
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm import (MoEMinimalLLM, load_and_cache_data, TextTokenDataset, MoEModelConfig)
from scaling import *
# -------------------------------
# Activations
# -------------------------------
class NonlinearityType(Enum):
    RELU = "relu"
    SILU = "silu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"

def get_activation_fn(name: str):
    if name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "swiglu":
        return lambda x: F.silu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:]
    elif name == "geglu":
        return lambda x: F.gelu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:]
    else:
        raise ValueError(f"Unknown activation {name}")

# -------------------------------
# Expert with configurable activation
# -------------------------------
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "silu", dropout: float = 0.1):
        super().__init__()
        self.activation_name = activation
        self.activation = get_activation_fn(activation)
        
        # For gated activations, need double dim
        if activation in ["swiglu", "geglu"]:
            assert d_ff % 2 == 0, "d_ff must be even for gated activations"
            inner_dim = d_ff * 2
        else:
            inner_dim = d_ff

        self.linear1 = nn.Linear(d_model, inner_dim, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear1(x)
        h = self.activation(h)
        h = self.dropout(h)
        return self.linear2(h)

# -------------------------------
# Training loop for activation benchmark
# -------------------------------
def train_moe_with_activation(config, train_loader, val_loader):
    print(f"\n🚀 Training with activation = {config.activation}")
    model = MoEMinimalLLM(config).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_losses, val_losses, val_steps = [], [], []

    for step in tqdm(range(1, config.max_steps + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, aux_loss = model(x, return_aux_loss=True)
            ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = ce_loss + (aux_loss if aux_loss is not None else 0.0)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        if step % config.eval_every == 0:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x, return_aux_loss=True)
                    val_loss += F.cross_entropy(
                        logits.view(-1, config.vocab_size), y.view(-1)
                    ).item()
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.numel()
            val_loss /= len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_steps.append(step)

            print(f"Step {step}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, acc={val_acc:.4f}")

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_steps": val_steps,
    }

# -------------------------------
# Benchmark function
# -------------------------------
def benchmark_activation(act_name: str, num_steps: int = 100):
    print(f"\n🔹 Benchmarking activation: {act_name}")

    config = MoEModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        num_experts=4,
        expert_top_k=2,
        num_documents=200,
        max_tokens=20000,
        max_seq_len=128,
        batch_size=16,
        max_steps=num_steps,
        eval_every=20,
    )
    config.activation = act_name

    texts, tokenizer, tokens = load_and_cache_data(config)
    config.vocab_size = len(tokenizer)
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    start = time.time()
    _, metrics = train_moe_with_activation(config, train_loader, val_loader)
    elapsed = time.time() - start

    return {
        "activation": act_name,
        "train_time_sec": elapsed,
        "time_per_step_sec": elapsed / num_steps,
        **metrics,
    }

# -------------------------------
# Plot results
# -------------------------------
def plot_loss_curves(results, save_path="activation_benchmark.png"):
    plt.figure(figsize=(10, 6))
    for r in results:
        steps = list(range(1, len(r["train_losses"]) + 1))
        plt.plot(steps, r["train_losses"], label=f"{r['activation']} (train)")
        if r["val_losses"]:
            plt.plot(r["val_steps"], r["val_losses"], linestyle="--", label=f"{r['activation']} (val)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Activation Function Benchmark (MoE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📉 Saved plot to {save_path}")


def main():
    # Load full dataset once
    texts, tokenizer, tokens = load_and_cache_data(MoEModelConfig())
    dataset = TextTokenDataset(tokens, 128)

    all_results = []
    for act in ["relu", "silu", "gelu", "swiglu", "geglu"]:
        res = benchmark_activation_scaling(
            act,
            dataset,
            tokenizer,
            dataset_fracs=[0.1, 0.3, 1.0]
        )
        all_results.extend(res)
        
    
    summary = fit_scaling_exponent(all_results)
    print("\n📊 Scaling Exponents")
    print(f"{'Activation':>10} | {'Exponent b':>10} | {'Irreducible c':>12}")
    print("-"*36)
    for act, p in summary.items():
        print(f"{act:>10} | {p['b']:10.3f} | {p['c']:12.3f}")


    return all_results

if __name__=='__main__':
    main()