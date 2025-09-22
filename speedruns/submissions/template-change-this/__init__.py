"""
T4 Speedrun Challenge Module
"""

from .config import T4SpeedrunConfig, get_t4_speedrun_config, create_custom_t4_config
from .speedrun import run_speedrun, SpeedrunTimer
from .leaderboard import SpeedrunLeaderboard

__version__ = "1.0.0"

__all__ = [
    "T4SpeedrunConfig",
    "get_t4_speedrun_config", 
    "create_custom_t4_config",
    "run_speedrun",
    "SpeedrunTimer",
    "SpeedrunLeaderboard",
]