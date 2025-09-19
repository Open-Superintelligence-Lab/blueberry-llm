
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Assuming the modified model is in auxiliary_loss_free_model
from auxiliary_loss_free_model import Transformer, ModelArgs

# --- 1. Task Definition: Transitive Reasoning ---
def generate_transitive_reasoning_data(num_samples=1000, vocab_size=20):
    """Generates data for the A>B, B>C => A?C task."""
    # Simple vocabulary: 0-9 for items, 10 for '>', 11 for ',', 12 for '?', 13 for padding
    # A > B
    # B > C
    # A ? C
    # For simplicity, let's use single tokens.
    # A, B, C will be random integers.
    # X: [A, 10, B, 11, B, 10, C, 11, A, 12, C] (len 11)
    # Y: [10] ('>')
    
    data = []
    labels = []
    
    for _ in range(num_samples):
        a, b, c = np.random.choice(range(10), 3, replace=False)
        
        # Ensure transitivity, e.g., A > B > C
        if not (a > b and b > c):
            # Simple sort to enforce order for this example
            a, b, c = sorted([a, b, c], reverse=True)

        # [A, >, B, ',', B, >, C, ',', A, ?, C]
        sequence = [a, 10, b, 11, b, 10, c, 11, a, 12, c]
        
        # The answer is always '>' (10)
        label = [10]

        data.append(sequence)
        labels.append(label)

    # Pad sequences to a fixed length if necessary, though here they are all equal
    return torch.LongTensor(data), torch.LongTensor(labels)

# --- 2. Experiment Setup ---
def run_experiment(dynamic_gate: bool, num_epochs=10):
    """
    Runs a single training experiment for a given gating strategy.
    
    Args:
        dynamic_gate (bool): If True, uses the dynamic load balancing gate.
        num_epochs (int): Number of training epochs.
        
    Returns:
        A dictionary containing results: accuracy, final_loss, expert_utilization.
    """
    torch.set_default_dtype(torch.bfloat16)
    print(f"--- Running Experiment with dynamic_gate = {dynamic_gate} ---")

    # Model Configuration
    args = ModelArgs(
        n_layers=4, # Smaller model for quick experiment
        n_heads=4,
        dim=128,
        n_routed_experts=8,
        n_activated_experts=2,
        vocab_size=20,
        max_seq_len=15,
        max_batch_size=32, # Match DataLoader batch_size
        dynamic_gate=dynamic_gate
    )
    
    model = Transformer(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Data
    X, Y = generate_transitive_reasoning_data(num_samples=2000)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # The model returns logits for the last token, which is what we want to predict
            logits = model(batch_x)
            
            # Target is the single label token
            loss = criterion(logits, batch_y.squeeze())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

    # --- 3. Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        X_test, Y_test = generate_transitive_reasoning_data(num_samples=500)
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        
        logits = model(X_test)
        predicted = torch.argmax(logits, dim=1)
        
        total = Y_test.size(0)
        correct = (predicted == Y_test.squeeze()).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # --- 4. Expert Utilization Analysis ---
    expert_utilization = {}
    if args.n_routed_experts > 0:
        for i, layer in enumerate(model.layers):
            if hasattr(layer.ffn, 'gate'):
                gate = layer.ffn.gate
                if hasattr(gate, 'expert_counts'):
                    counts = gate.expert_counts.cpu().numpy()
                    expert_utilization[f'layer_{i}'] = counts / counts.sum() if counts.sum() > 0 else counts
                    print(f"Expert utilization for layer {i}: {expert_utilization[f'layer_{i}']}")

    return {
        "accuracy": accuracy,
        "final_loss": total_loss / len(loader),
        "expert_utilization": expert_utilization
    }

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Run both experiments
    baseline_results = run_experiment(dynamic_gate=False)
    dynamic_results = run_experiment(dynamic_gate=True)

    # --- 6. Comparison ---
    print("\n" + "="*50)
    print("              EXPERIMENT COMPARISON")
    print("="*50)
    print(f"Baseline (Static Gate) Accuracy: {baseline_results['accuracy']:.2f}%")
    print(f"Dynamic Gate Accuracy:           {dynamic_results['accuracy']:.2f}%")
    print("\n" + "-"*50 + "\n")

    # Plot expert utilization
    for layer_name, utilization in dynamic_results['expert_utilization'].items():
        if utilization is not None:
            plt.figure(figsize=(12, 5))
            plt.suptitle(f"Expert Utilization Comparison for {layer_name}")

            # Dynamic
            plt.subplot(1, 2, 1)
            plt.bar(range(len(utilization)), utilization)
            plt.title("Dynamic Gate")
            plt.xlabel("Expert ID")
            plt.ylabel("Proportion of Tokens")
            plt.ylim(0, 1)

            # Baseline (if available)
            if layer_name in baseline_results['expert_utilization']:
                 plt.subplot(1, 2, 2)
                 baseline_util = baseline_results['expert_utilization'][layer_name]
                 plt.bar(range(len(baseline_util)), baseline_util, color='orange')
                 plt.title("Baseline (Static Gate)")
                 plt.xlabel("Expert ID")
                 plt.ylim(0, 1)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"{layer_name}_utilization_comparison.png")
            print(f"Saved utilization plot to {layer_name}_utilization_comparison.png")

    print("\n" + "="*50)
