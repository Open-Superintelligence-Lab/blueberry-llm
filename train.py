#!/usr/bin/env python3
"""
Entry point for Blueberry LLM training.
This script redirects to the core training functionality with proper argument passing.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the main training script
if __name__ == "__main__":
    # Import the train_auto module and run it directly
    # This ensures command line arguments are properly parsed
    import core.train_auto
    core.train_auto.main()
