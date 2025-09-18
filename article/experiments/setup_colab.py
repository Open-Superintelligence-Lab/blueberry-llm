#!/usr/bin/env python3
"""
Setup script for Google Colab environment
Run this first to set up the environment properly
"""

import os
import sys
import subprocess

def setup_colab_environment():
    """Set up the Colab environment for AMP experiments"""
    print("🔧 Setting up Colab environment for AMP experiments...")
    
    # Check if we're in Colab
    try:
        import google.colab
        in_colab = True
        print("✅ Running in Google Colab")
    except ImportError:
        in_colab = False
        print("ℹ️  Running locally (not in Colab)")
    
    # Install required packages
    print("\n📦 Installing required packages...")
    packages = [
        "matplotlib",
        "seaborn", 
        "pandas",
        "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Set up paths
    print("\n📁 Setting up Python paths...")
    
    # Get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the right location
    if "blueberry-llm" in cwd:
        print("✅ In blueberry-llm directory")
        project_root = cwd
    else:
        # Try to find blueberry-llm directory
        if in_colab:
            # In Colab, assume it's in /content
            project_root = "/content/blueberry-llm"
            if os.path.exists(project_root):
                os.chdir(project_root)
                print(f"✅ Changed to project root: {project_root}")
            else:
                print("❌ Cannot find blueberry-llm directory")
                print("Please upload the blueberry-llm repository to Colab")
                return False
        else:
            print("❌ Please run this script from the blueberry-llm directory")
            return False
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ Added {project_root} to Python path")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available")
        
        # Test blueberry-llm imports
        from core.auto_config import AutoConfig
        print("✅ Core imports working")
        
        from legacy.llm import MoEModelConfig
        print("✅ Legacy imports working")
        
        from data.dataset import TextTokenDataset
        print("✅ Data imports working")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Setup complete! Ready to run experiments.")
    print("\nNext steps:")
    print("1. Run: python article/experiments/quick_test.py")
    print("2. If test passes, run: python article/experiments/amp_experiment_runner.py --full")
    print("3. Analyze results: python article/experiments/analyze_results.py --all")
    
    return True

if __name__ == "__main__":
    success = setup_colab_environment()
    if not success:
        sys.exit(1)
