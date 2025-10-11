# Inference Guide for Trained Gated DeltaNet Models

## Overview

After training, your model is saved in the `checkpoints/` directory. You can load and use it for text generation or other tasks.

## Model Files

Training produces two checkpoint files:

1. **`checkpoints/best_model.pt`** - Model with best validation loss (recommended for inference)
2. **`checkpoints/final_model.pt`** - Model after all training steps

Each checkpoint contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state (for resuming training)
- `scheduler_state_dict` - LR scheduler state
- `global_step` - Training step count
- `best_val_loss` - Best validation loss achieved
- `config` - Full model configuration

## Quick Inference

### Run the provided inference script:

```bash
cd /root/blueberry-llm/experiments/exp6_gated_deltanet_training
python inference.py
```

This will:
- Load the best model checkpoint
- Generate text from example prompts
- Show how to use the model

## Manual Inference (Python Code)

### 1. Load the Model

```python
import torch
from pathlib import Path
from experiments.exp6_gated_deltanet_training.models import GatedDeltaNetWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
checkpoint_path = Path("experiments/exp6_gated_deltanet_training/checkpoints/best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Create and load model
config = checkpoint['config']
model = GatedDeltaNetWrapper(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded! Trained for {checkpoint['global_step']} steps")
```

### 2. Load Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
```

### 3. Generate Text

```python
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Generate
with torch.no_grad():
    for _ in range(50):  # Generate 50 tokens
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Get next token (greedy decoding)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

# Decode
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(generated_text)
```

## Using with Hugging Face

The model is compatible with Hugging Face's `generate()` API:

```python
# The underlying FLA model supports generate
outputs = model.model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Resume Training

To continue training from a checkpoint:

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# Create model and optimizer
model = GatedDeltaNetWrapper(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from checkpoint['global_step']
```

## Export for Production

### Save just the model weights:

```python
# Lighter weight file (just model, no optimizer)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'model_weights_only.pt')
```

### Convert to Hugging Face format:

```python
# The FLA model can be saved in HF format
model.model.save_pretrained('my_deltanet_model')
tokenizer.save_pretrained('my_deltanet_model')

# Later, load it:
from fla.models import DeltaNetForCausalLM
model = DeltaNetForCausalLM.from_pretrained('my_deltanet_model')
```

## Performance Tips

### 1. Use bfloat16 for inference:

```python
model = model.to(dtype=torch.bfloat16)  # 2x faster, half memory
```

### 2. Batch inference:

```python
# Process multiple prompts at once
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids.to(device)
outputs = model(input_ids)
```

### 3. Use torch.compile (PyTorch 2.0+):

```python
model = torch.compile(model)  # JIT compilation for speed
```

## Common Issues

### "Checkpoint not found"
Make sure you've completed training:
```bash
python run_experiment.py
```

### OOM during inference
Reduce batch size or use bfloat16:
```python
model = model.to(dtype=torch.bfloat16)
```

### Slow generation
- Use GPU if available
- Enable torch.compile()
- Use bfloat16 precision
- Batch multiple prompts

### Poor quality generations
Your model may need more training:
- Increase `max_steps` in config
- Use more training data
- Try larger model size

## Evaluation

Evaluate your model on benchmarks:

```python
# Perplexity on test set
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset

# Load test data
texts, tokenizer, tokens = load_and_cache_data(config)
test_dataset = TextTokenDataset(tokens, config.max_seq_len)

# Calculate perplexity
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch, labels=batch)
        total_loss += outputs.loss.item()

perplexity = torch.exp(torch.tensor(total_loss / len(test_loader)))
print(f"Perplexity: {perplexity:.2f}")
```

## Next Steps

- Fine-tune on specific domains
- Evaluate on downstream tasks
- Deploy as API endpoint
- Experiment with generation parameters
- Try different decoding strategies (beam search, nucleus sampling)

## References

- FLA Documentation: https://github.com/fla-org/flash-linear-attention
- Hugging Face Generation Guide: https://huggingface.co/docs/transformers/generation_strategies
- Training Guide: See `README.md` in this directory

