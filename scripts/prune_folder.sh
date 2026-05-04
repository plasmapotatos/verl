#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PARENT_DIR [--dry-run] [--mode keep-latest|all|aggressive]" >&2
  exit 1
fi

parent_dir=$1
shift

if [[ ! -d "$parent_dir" ]]; then
  echo "Error: directory does not exist: $parent_dir" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_script="$script_dir/prune_project_checkpoints.sh"

if [[ ! -f "$project_script" ]]; then
  echo "Error: project prune script not found: $project_script" >&2
  exit 1
fi

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      extra_args+=("$1")
      shift
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Error: --mode requires a value (keep-latest|all|aggressive)" >&2
        exit 1
      fi
      extra_args+=("$1" "$2")
      shift 2
      ;;
    --mode=*)
      extra_args+=("$1")
      shift
      ;;
    *)
      echo "Error: unsupported argument: $1" >&2
      exit 1
      ;;
  esac
done

found_any=0
for project_dir in "$parent_dir"/*; do
  [[ -d "$project_dir" ]] || continue

  echo "=== Pruning project: $project_dir ==="
  if bash "$project_script" "$project_dir" "${extra_args[@]}"; then
    found_any=1
  else
    echo "Skipped (no experiments with checkpoints): $project_dir"
  fi
  echo
done

if [[ $found_any -eq 0 ]]; then
  echo "No project directories with checkpoints found under $parent_dir" >&2
  exit 1
fi
