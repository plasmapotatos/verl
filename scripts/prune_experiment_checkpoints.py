#!/usr/bin/env python3
"""
Checkpoint garbage collector for VERL RL training.

Given a project/experiment directory, this script:
- Finds all global_step_* directories
- Keeps the latest checkpoint (largest step number) completely intact unless --all is set
- Deletes actor/ folders from pruned checkpoints to save space
- Deletes top-level .pt files from pruned checkpoints
- Preserves merged_hf_model/ and generations/ folders for inference and analysis
"""

import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser(
        description='Prune actor folders and top-level .pt files from checkpoints'
    )
    parser.add_argument('experiment_dir', help='Path to experiment directory containing global_step_* checkpoint dirs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--all', action='store_true',
                        help='Also prune the latest checkpoint instead of keeping it intact')
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist")
        return
    
    # Find all valid steps
    steps = []
    for item in os.listdir(experiment_dir):
        step_path = os.path.join(experiment_dir, item)
        if item.startswith('global_step_') and os.path.isdir(step_path):
            try:
                step_num = int(item.split('_')[-1])
                steps.append((step_num, item))
            except (ValueError, IndexError):
                continue
    
    if not steps:
        print("No valid checkpoint steps found")
        return
    
    # Find latest step (largest step number)
    latest_step = max(steps, key=lambda x: x[0])
    
    print(f"Found {len(steps)} valid checkpoint steps")
    if args.all:
        print(f"Pruning all checkpoints, including latest step: {latest_step[1]} (step {latest_step[0]})")
        to_delete = [s[1] for s in steps]
    else:
        print(f"Keeping latest step: {latest_step[1]} (step {latest_step[0]})")
        keep_dirs = {latest_step[1]}
        to_delete = [s[1] for s in steps if s[1] not in keep_dirs]
    
    if not to_delete:
        print("No checkpoints to delete")
        return
    
    print(f"Will delete actor folders and top-level .pt files from {len(to_delete)} checkpoints:")
    for d in to_delete:
        print(f"  - {d}")
    
    if args.dry_run:
        print("\nDry run - no files deleted")
    else:
        for dirname in to_delete:
            full_path = os.path.join(experiment_dir, dirname)
            try:
                deleted_any = False

                # Delete actor folder if present.
                actor_path = os.path.join(full_path, 'actor')
                if os.path.exists(actor_path):
                    shutil.rmtree(actor_path)
                    print(f"Deleted actor from {dirname}")
                    deleted_any = True

                # Delete only .pt files directly inside the checkpoint directory.
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isfile(item_path) and item.endswith('.pt'):
                        os.remove(item_path)
                        print(f"Deleted {item} from {dirname}")
                        deleted_any = True

                if not deleted_any:
                    print(f"No actor folder or top-level .pt files found in {dirname}")

            except Exception as e:
                print(f"Error pruning {dirname}: {e}")


if __name__ == '__main__':
    main()
