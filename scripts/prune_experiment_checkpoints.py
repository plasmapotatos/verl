#!/usr/bin/env python3
"""
Checkpoint garbage collector for VERL RL training.

Given a project/experiment directory, this script:
- Finds all global_step_* directories
- Deletes actor/ folders and top-level .pt files based on --mode:
    keep-latest (default): prune all steps except the latest
    all:                   prune every step (including latest), keep merged_hf_model/
    aggressive:            prune every step AND delete merged_hf_model/ folders
- Always preserves generations/ and other analysis files
"""

import os
import argparse
import shutil


MODES = ('keep-latest', 'all', 'aggressive')


def main():
    parser = argparse.ArgumentParser(
        description='Prune actor folders and top-level .pt files from checkpoints'
    )
    parser.add_argument('experiment_dir', help='Path to experiment directory containing global_step_* checkpoint dirs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--mode', choices=MODES, default='keep-latest',
                        help=('keep-latest: keep latest step intact, prune older. '
                              'all: prune every step (incl. latest), keep merged_hf_model. '
                              'aggressive: prune every step AND delete merged_hf_model.'))
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
    if args.mode == 'aggressive':
        print(f"AGGRESSIVE: pruning all checkpoints and deleting merged_hf_model/ folders. Latest step: {latest_step[1]} (step {latest_step[0]})")
        to_delete = [s[1] for s in steps]
    elif args.mode == 'all':
        print(f"ALL: pruning all checkpoints (keeping merged_hf_model/). Latest step: {latest_step[1]} (step {latest_step[0]})")
        to_delete = [s[1] for s in steps]
    else:
        print(f"KEEP-LATEST: keeping latest step intact: {latest_step[1]} (step {latest_step[0]})")
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

                # In aggressive mode, also delete merged_hf_model/ folders.
                if args.mode == 'aggressive':
                    merged_path = os.path.join(full_path, 'merged_hf_model')
                    if os.path.exists(merged_path):
                        shutil.rmtree(merged_path)
                        print(f"Deleted merged_hf_model from {dirname}")
                        deleted_any = True

                if not deleted_any:
                    print(f"Nothing to prune in {dirname}")

            except Exception as e:
                print(f"Error pruning {dirname}: {e}")


if __name__ == '__main__':
    main()
