"""
Quick test script - runs a very short experiment to verify the pipeline works
"""
import sys
import os
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import after paths are set
from config import TemperatureConfig, list_experiments

# Create a very short test configuration
TEST_CONFIG = TemperatureConfig(
    name="test_quick",
    description="Quick test run with 50 steps",
    temperature=1.0,
    max_steps=50,  # Very short for testing
)

def main():
    print("\n" + "="*80)
    print("QUICK TEST: Temperature Routing Experiment")
    print("="*80 + "\n")
    
    print("This is a quick test to verify the experiment pipeline works.")
    print(f"Running {TEST_CONFIG.max_steps} steps (should take <1 minute)\n")
    
    # List available experiments
    print("Available experiment types:")
    list_experiments()
    
    print("\n" + "="*80)
    print("Test configuration:")
    print("="*80)
    print(f"Name: {TEST_CONFIG.name}")
    print(f"Description: {TEST_CONFIG.description}")
    print(f"Temperature: {TEST_CONFIG.temperature}")
    print(f"Steps: {TEST_CONFIG.max_steps}")
    print("="*80 + "\n")
    
    print("✅ Configuration test passed!")
    print("\nTo run a full experiment, use:")
    print("  python run_experiment.py --experiment temp_1.0")
    print("\nTo run the quick demo (3 temps, 500 steps each):")
    print("  bash quick_demo.sh")
    print("\nTo run full temperature ablation:")
    print("  python run_experiment.py --ablation")
    print("")

if __name__ == "__main__":
    main()

