import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'flash-muon'))

from flash_muon import Muon as FlashMuon
from .muon import Muon, zeropower_via_newtonschulz5

__all__ = ["FlashMuon", "Muon", "zeropower_via_newtonschulz4"]
