#!/usr/bin/env python3
"""
Checkpoint garbage collector for VERL RL training.

Given a project/experiment directory, this script:
- Finds all global_step_* directories
- Evaluates each step's average accuracy from generation eval files
- Keeps the highest-scoring checkpoint and the last checkpoint
- Deletes model files (actor/, merged_hf_model/, data.pt) from other checkpoints (with --dry-run option for safety)
"""

import os
import json
import argparse
import shutil


def get_accuracy_for_step(step_dir):
    """
    Get average accuracy for a checkpoint step from its eval files.
    
    Args:
        step_dir: Path to global_step_* directory
        
    Returns:
        Average accuracy across all eval files, or None if no valid evals found
    """
    gen_dir = os.path.join(step_dir, 'generations')
    if not os.path.exists(gen_dir):
        return None
    
    accuracies = []
    for filename in os.listdir(gen_dir):
        if filename.endswith('_eval.json'):
            filepath = os.path.join(gen_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if 'metrics' in data and 'accuracy' in data['metrics'] and isinstance(data['metrics']['accuracy'], (int, float)):
                        accuracies.append(data['metrics']['accuracy'])
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue
    
    if not accuracies:
        return None
    
    return sum(accuracies) / len(accuracies)


def main():
    parser = argparse.ArgumentParser(description='Prune model files from checkpoints, keeping best and last')
    parser.add_argument('experiment_dir', help='Path to experiment directory containing global_step_* checkpoint dirs')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist")
        return
    
    # Find all valid steps with accuracy
    steps = []
    for item in os.listdir(experiment_dir):
        step_path = os.path.join(experiment_dir, item)
        if item.startswith('global_step_') and os.path.isdir(step_path):
            try:
                step_num = int(item.split('_')[-1])
                acc = get_accuracy_for_step(step_path)
                if acc is not None:
                    steps.append((step_num, acc, item))
            except (ValueError, IndexError):
                continue
    
    if not steps:
        print("No valid checkpoint steps found with accuracy metrics")
        return
    
    # Find best and last steps
    best_step = max(steps, key=lambda x: x[1])
    last_step = max(steps, key=lambda x: x[0])
    
    keep_dirs = set([best_step[2], last_step[2]])
    
    print(f"Found {len(steps)} valid checkpoint steps")
    print(f"Keeping best accuracy: {best_step[2]} (avg acc: {best_step[1]:.4f})")
    print(f"Keeping last step: {last_step[2]} (avg acc: {last_step[1]:.4f})")
    
    to_delete = [s[2] for s in steps if s[2] not in keep_dirs]
    
    if not to_delete:
        print("No checkpoints to delete")
        return
    
    print(f"Will delete model files from {len(to_delete)} checkpoints:")
    for d in to_delete:
        print(f"  - {d}")
    
    if args.dry_run:
        print("\nDry run - no files deleted")
    else:
        confirm = input(f"\nDelete model files from {len(to_delete)} checkpoints? (y/N): ")
        if confirm.lower() == 'y':
            for dirname in to_delete:
                full_path = os.path.join(experiment_dir, dirname)
                try:
                    # Delete actor folder
                    actor_path = os.path.join(full_path, 'actor')
                    if os.path.exists(actor_path):
                        shutil.rmtree(actor_path)
                        print(f"Deleted actor from {dirname}")
                    
                    # Delete merged_hf_model folder
                    merged_path = os.path.join(full_path, 'merged_hf_model')
                    if os.path.exists(merged_path):
                        shutil.rmtree(merged_path)
                        print(f"Deleted merged_hf_model from {dirname}")
                    
                    # Delete data.pt file
                    data_pt_path = os.path.join(full_path, 'data.pt')
                    if os.path.exists(data_pt_path):
                        os.remove(data_pt_path)
                        print(f"Deleted data.pt from {dirname}")
                        
                except Exception as e:
                    print(f"Error deleting files from {dirname}: {e}")
        else:
            print("Aborted")


if __name__ == '__main__':
    main()