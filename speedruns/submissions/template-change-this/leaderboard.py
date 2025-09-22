#!/usr/bin/env python3
"""
T4 Speedrun Leaderboard
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


class SpeedrunLeaderboard:
    """Simple leaderboard for T4 speedrun challenge"""
    
    def __init__(self, leaderboard_file: str = "speedrun_leaderboard.json"):
        self.leaderboard_file = Path(__file__).parent / "results" / leaderboard_file
        self.leaderboard_file.parent.mkdir(exist_ok=True)
        self.entries = self.load_leaderboard()
    
    def load_leaderboard(self) -> List[Dict[str, Any]]:
        """Load leaderboard from file"""
        if self.leaderboard_file.exists():
            with open(self.leaderboard_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_leaderboard(self):
        """Save leaderboard to file"""
        with open(self.leaderboard_file, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(self, results: Dict[str, Any], participant_name: str = None):
        """Add a new entry to the leaderboard"""
        if not results.get('completed', False):
            print("Cannot add incomplete speedrun to leaderboard")
            return False
        
        entry = {
            'participant': participant_name or "Anonymous",
            'final_val_loss': results['final_val_loss'],
            'total_time_minutes': results['total_time_minutes'],
        }
        
        self.entries.append(entry)
        self.sort_leaderboard()
        self.save_leaderboard()
        
        print(f"Added: {entry['participant']} - Val Loss: {entry['final_val_loss']:.6f}")
        return True
    
    def sort_leaderboard(self):
        """Sort leaderboard by validation loss (ascending)"""
        self.entries.sort(key=lambda x: x['final_val_loss'])
    
    def print_leaderboard(self, top_n: int = 10):
        """Print formatted leaderboard"""
        print("\nT4 SPEEDRUN LEADERBOARD")
        print("=" * 50)
        
        if not self.entries:
            print("No entries yet")
            return
        
        top_entries = self.entries[:top_n]
        
        print(f"{'Rank':<4} {'Participant':<15} {'Val Loss':<12} {'Time':<8}")
        print("-" * 50)
        
        for i, entry in enumerate(top_entries, 1):
            time_str = f"{entry['total_time_minutes']:.1f}m"
            print(f"{i:<4} {entry['participant']:<15} {entry['final_val_loss']:<12.6f} {time_str:<8}")
        
        print(f"Total entries: {len(self.entries)}")


def load_results_from_file(results_file: str) -> Dict[str, Any]:
    """Load results from a JSON file"""
    results_path = Path(__file__).parent / "results" / results_file
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="T4 Speedrun Leaderboard")
    parser.add_argument("--show", action="store_true", help="Show leaderboard")
    parser.add_argument("--add", type=str, help="Add results file to leaderboard")
    parser.add_argument("--participant", type=str, help="Participant name")
    parser.add_argument("--top", type=int, default=10, help="Number of top entries to show")
    
    args = parser.parse_args()
    
    leaderboard = SpeedrunLeaderboard()
    
    if args.add:
        results = load_results_from_file(args.add)
        if results:
            leaderboard.add_entry(results, args.participant)
    else:
        leaderboard.print_leaderboard(args.top)


if __name__ == "__main__":
    main()