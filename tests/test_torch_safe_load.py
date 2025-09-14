import torch


# Define a simple class at module scope so it is picklable
class DummyTok:
    def __init__(self):
        self.name = "dummy"


def test_weights_only_blocks_unallowlisted_classes(tmp_path):
    # Prepare a checkpoint that contains a non-allowlisted Python object
    ckpt_path = tmp_path / "ckpt_with_dummy.pt"
    torch.save({"tokenizer": DummyTok(), "x": 1}, ckpt_path)

    # Attempt to load with weights_only=True should fail with a safe unpickler error
    try:
        torch.load(ckpt_path, weights_only=True)
        assert False, "torch.load(weights_only=True) should have failed for unallowlisted class"
    except Exception as e:
        msg = str(e)
        assert (
            "Unsupported global:" in msg or "Weights only load failed" in msg
        ), f"Unexpected error message: {msg}"


def test_add_safe_globals_allows_loading(tmp_path):
    ckpt_path = tmp_path / "ckpt_with_dummy_allowed.pt"
    torch.save({"tokenizer": DummyTok(), "x": 1}, ckpt_path)

    # Allowlist DummyTok and load with weights_only=True
    torch.serialization.add_safe_globals([DummyTok])
    obj = torch.load(ckpt_path, weights_only=True)
    assert isinstance(obj["tokenizer"], DummyTok)


def test_weights_only_false_loads_pickle_normally(tmp_path):
    ckpt_path = tmp_path / "ckpt_pickle.pt"
    torch.save({"tokenizer": DummyTok(), "x": 1}, ckpt_path)

    # Using weights_only=False should load without the safe unpickler
    obj = torch.load(ckpt_path, weights_only=False)
    assert isinstance(obj["tokenizer"], DummyTok)

