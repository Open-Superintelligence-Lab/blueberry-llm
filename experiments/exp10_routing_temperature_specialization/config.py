"""
Configuration for routing temperature experiments
"""
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TemperatureConfig:
    """Configuration for a single temperature experiment"""
    name: str
    description: str
    
    # Temperature settings
    temperature: float = 1.0
    temperature_schedule: Optional[Literal["linear", "cosine", "exponential", "step"]] = None
    temperature_start: Optional[float] = None
    temperature_end: Optional[float] = None
    
    # Training settings (inherit from MoEModelConfig defaults)
    max_steps: int = 500
    
    def get_temperature_at_step(self, step: int) -> float:
        """Calculate temperature at given training step"""
        if self.temperature_schedule is None:
            return self.temperature
        
        # Use schedule
        start_temp = self.temperature_start if self.temperature_start is not None else self.temperature
        end_temp = self.temperature_end if self.temperature_end is not None else self.temperature
        
        progress = step / self.max_steps
        
        if self.temperature_schedule == "linear":
            return start_temp + (end_temp - start_temp) * progress
        
        elif self.temperature_schedule == "cosine":
            import math
            return end_temp + (start_temp - end_temp) * 0.5 * (1 + math.cos(math.pi * progress))
        
        elif self.temperature_schedule == "exponential":
            import math
            # Exponential decay: temp = start * (end/start)^progress
            return start_temp * (end_temp / start_temp) ** progress
        
        elif self.temperature_schedule == "step":
            # Step schedule: 5.0 (0-100) → 2.0 (100-300) → 1.0 (300+)
            if step < 100:
                return 5.0
            elif step < 300:
                return 2.0
            else:
                return 1.0
        
        return self.temperature


# Temperature Ablation Experiments
TEMPERATURE_ABLATION = {
    "temp_0.5": TemperatureConfig(
        name="temp_0.5",
        description="Very sharp routing (strong exploitation)",
        temperature=0.5,
    ),
    "temp_0.7": TemperatureConfig(
        name="temp_0.7",
        description="Sharp routing (moderate exploitation)",
        temperature=0.7,
    ),
    "temp_1.0": TemperatureConfig(
        name="temp_1.0",
        description="Standard softmax (baseline)",
        temperature=1.0,
    ),
    "temp_1.5": TemperatureConfig(
        name="temp_1.5",
        description="Slightly softer routing",
        temperature=1.5,
    ),
    "temp_2.0": TemperatureConfig(
        name="temp_2.0",
        description="Softer routing (more exploration)",
        temperature=2.0,
    ),
    "temp_3.0": TemperatureConfig(
        name="temp_3.0",
        description="Soft routing (high exploration)",
        temperature=3.0,
    ),
    "temp_5.0": TemperatureConfig(
        name="temp_5.0",
        description="Very soft routing (maximum exploration)",
        temperature=5.0,
    ),
    "temp_10.0": TemperatureConfig(
        name="temp_10.0",
        description="Nearly uniform routing (extreme exploration)",
        temperature=10.0,
    ),
}

# Temperature Scheduling Experiments
TEMPERATURE_SCHEDULES = {
    "schedule_linear": TemperatureConfig(
        name="schedule_linear",
        description="Linear decay from 5.0 → 1.0",
        temperature=5.0,
        temperature_schedule="linear",
        temperature_start=5.0,
        temperature_end=1.0,
    ),
    "schedule_cosine": TemperatureConfig(
        name="schedule_cosine",
        description="Cosine decay from 5.0 → 1.0",
        temperature=5.0,
        temperature_schedule="cosine",
        temperature_start=5.0,
        temperature_end=1.0,
    ),
    "schedule_exp": TemperatureConfig(
        name="schedule_exp",
        description="Exponential decay from 5.0 → 1.0",
        temperature=5.0,
        temperature_schedule="exponential",
        temperature_start=5.0,
        temperature_end=1.0,
    ),
    "schedule_step": TemperatureConfig(
        name="schedule_step",
        description="Step decay: 5.0 (0-100) → 2.0 (100-300) → 1.0 (300+)",
        temperature=5.0,
        temperature_schedule="step",
    ),
}

# Extended training with best temperature
EXTENDED_TRAINING = {
    "temp_best_long": TemperatureConfig(
        name="temp_best_long",
        description="Best temperature from ablation, trained for 1000 steps",
        temperature=2.0,  # Will be updated after ablation
        max_steps=1000,
    ),
}

# All experiments
ALL_EXPERIMENTS = {
    **TEMPERATURE_ABLATION,
    **TEMPERATURE_SCHEDULES,
    **EXTENDED_TRAINING,
}


def get_experiment_config(name: str) -> TemperatureConfig:
    """Get experiment configuration by name"""
    if name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(ALL_EXPERIMENTS.keys())}")
    return ALL_EXPERIMENTS[name]


def list_experiments():
    """Print all available experiments"""
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)
    
    print("\n📊 TEMPERATURE ABLATION (500 steps)")
    print("-" * 80)
    for name, config in TEMPERATURE_ABLATION.items():
        print(f"  {name:20s} - Temp: {config.temperature:5.1f} - {config.description}")
    
    print("\n📈 TEMPERATURE SCHEDULES (500 steps)")
    print("-" * 80)
    for name, config in TEMPERATURE_SCHEDULES.items():
        schedule_desc = f"{config.temperature_start} → {config.temperature_end}" if config.temperature_schedule else "constant"
        print(f"  {name:20s} - Schedule: {config.temperature_schedule or 'none':12s} - {config.description}")
    
    print("\n🔬 EXTENDED TRAINING (1000 steps)")
    print("-" * 80)
    for name, config in EXTENDED_TRAINING.items():
        print(f"  {name:20s} - Temp: {config.temperature:5.1f} - {config.description}")
    
    print("\n" + "="*80)
    print(f"Total: {len(ALL_EXPERIMENTS)} experiments")
    print("="*80 + "\n")

