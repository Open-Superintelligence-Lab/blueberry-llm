import types
import torch
import sys
from pathlib import Path

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_stub_tokenizer():
    class StubTok:
        def __init__(self):
            self.vocab_size = 128
            self.eos_token = "<eos>"
            self.eos_token_id = 0
            self.pad_token = None

        def encode(self, text, return_tensors=None, add_special_tokens=False):
            # simple byte-level fallback
            ids = [min(127, ord(c)) for c in text]
            return torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids

        def decode(self, idx):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            if isinstance(idx, list):
                return "".join(chr(int(i)) for i in idx)
            return chr(int(idx))

    return StubTok()


def test_load_model_creates_tokenizer_when_missing(monkeypatch):
    from llm import MoEMinimalLLM, MoEModelConfig
    import inference

    # Minimal model config
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

    # Build a tiny model and state dict
    model = MoEMinimalLLM(cfg)
    state = model.state_dict()

    # Fake checkpoint returned by torch.load (without 'tokenizer')
    ckpt = {"model_state_dict": state, "config": cfg}

    # Monkeypatch torch.load to avoid filesystem and unpickling issues
    monkeypatch.setattr(inference.torch, "load", lambda *a, **k: ckpt)

    # Stub out AutoTokenizer.from_pretrained to avoid network
    created = {"called": False, "args": None}
    stub = make_stub_tokenizer()

    def fake_from_pretrained(name, *a, **k):
        created["called"] = True
        created["args"] = (name, a, k)
        return stub

    monkeypatch.setattr(inference, "AutoTokenizer", types.SimpleNamespace(from_pretrained=fake_from_pretrained))

    loaded_model, tok, device, loaded_cfg = inference.load_model("dummy.pt")

    assert created["called"] is True
    assert tok is stub
    assert isinstance(loaded_model, MoEMinimalLLM)
    assert loaded_cfg.vocab_size == cfg.vocab_size
    assert str(device) in ("cpu", "cuda")
