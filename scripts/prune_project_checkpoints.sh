#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PROJECT_DIR [--dry-run] [--all]" >&2
  exit 1
fi

project_dir=$1
shift

if [[ ! -d "$project_dir" ]]; then
  echo "Error: directory does not exist: $project_dir" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prune_script="$script_dir/prine_experiment_checkpoints.py"

if [[ ! -f "$prune_script" ]]; then
  echo "Error: prune script not found: $prune_script" >&2
  exit 1
fi

found_any=0
extra_args=()

for arg in "$@"; do
  case "$arg" in
    --dry-run|--all)
      extra_args+=("$arg")
      ;;
    *)
      echo "Error: unsupported argument: $arg" >&2
      exit 1
      ;;
  esac
done

for experiment_dir in "$project_dir"/*; do
  [[ -d "$experiment_dir" ]] || continue

  if compgen -G "$experiment_dir/global_step_*" > /dev/null; then
    found_any=1
    echo "Pruning experiment: $experiment_dir"
    python3 "$prune_script" "$experiment_dir" "${extra_args[@]}"
    echo
  fi
done

if [[ $found_any -eq 0 ]]; then
  echo "No experiment directories with global_step_* checkpoints found under $project_dir" >&2
  exit 1
fi
