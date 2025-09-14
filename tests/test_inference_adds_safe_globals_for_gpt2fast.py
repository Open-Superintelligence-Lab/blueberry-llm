import sys
from pathlib import Path
import types


# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_load_model_allowlists_gpt2_tokenizer(monkeypatch):
    import inference
    from llm import MoEModelConfig, MoEMinimalLLM
    from transformers import GPT2TokenizerFast

    # Build a tiny, consistent checkpoint
    cfg = MoEModelConfig(
        vocab_size=128,
        max_seq_len=8,
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_ff=256,
        num_experts=2,
        batch_size=1,
        max_steps=1,
    )
    model = MoEMinimalLLM(cfg)
    ckpt = {"model_state_dict": model.state_dict(), "config": cfg}

    # Capture safe globals calls
    called = {"classes": []}

    def fake_add_safe(globs):
        called["classes"].extend(globs)

    monkeypatch.setattr(inference.torch.serialization, "add_safe_globals", fake_add_safe)
    # Avoid actual file I/O
    monkeypatch.setattr(inference.torch, "load", lambda *a, **k: ckpt)
    # Avoid tokenizer download
    monkeypatch.setattr(
        inference,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(
            eos_token="<eos>", eos_token_id=0, pad_token=None, vocab_size=128, encode=lambda *x, **y: [0], decode=lambda x: ""
        )),
    )

    inference.load_model("dummy.pt")

    # Expect GPT2TokenizerFast to be in the allowlist
    assert any(cls is GPT2TokenizerFast for cls in called["classes"]), (
        "inference.load_model should add GPT2TokenizerFast to torch.serialization.add_safe_globals"
    )

