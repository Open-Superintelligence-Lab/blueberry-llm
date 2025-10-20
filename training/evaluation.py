import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.amp import autocast
from configs.moe_config import MoEModelConfig
from utils.helpers import unpack_batch

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: MoEModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = unpack_batch(batch, device)

            with autocast('cuda', dtype=torch.float16, enabled=config.use_amp):
                # MoE model evaluation
                logits = model(x, return_aux_loss=False)  # Don't return aux loss during eval
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}
