#!/bin/bash
# queue_bench.sh — measure SLURM queue wait time vs (wall-time limit, GPUs/node).
#
# Each probe job is a no-op: it records its own start timestamp and exits. We
# log submit_ts up front; start_ts comes from sacct once the scheduler runs it.
# Run `submit` now, walk away, then `report` later (or from another session).
#
# Usage:
#   ./queue_bench.sh submit [--label LABEL] [--times "30 60 120 240"] [--gpus "1 2 4"]
#   ./queue_bench.sh report <run-dir>
#   ./queue_bench.sh list
#   ./queue_bench.sh cancel <run-dir>
#
# Example (run one now, another at midnight):
#   ./queue_bench.sh submit --label afternoon
#   # later:
#   ./queue_bench.sh submit --label midnight
#   ./queue_bench.sh report queue_bench/afternoon_*
#   ./queue_bench.sh report queue_bench/midnight_*

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BENCH_ROOT="${SELF_DIR}/queue_bench"

SBATCH_COMMON=(
    --account=bbsg-dtai-gh
    --partition=ghx4
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task=1
    --mem=0
)

usage() {
    sed -n '2,20p' "$0" >&2
    exit 1
}

fmt_time() {
    # minutes -> HH:MM:SS
    local m="$1"
    printf '%02d:%02d:00' "$((m / 60))" "$((m % 60))"
}

cmd_submit() {
    local label="run"
    local times="30 60 120 240"
    local gpus_list="1 2 4"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --label) label="$2"; shift 2 ;;
            --label=*) label="${1#--label=}"; shift ;;
            --times) times="$2"; shift 2 ;;
            --times=*) times="${1#--times=}"; shift ;;
            --gpus) gpus_list="$2"; shift 2 ;;
            --gpus=*) gpus_list="${1#--gpus=}"; shift ;;
            -h|--help) usage ;;
            *) echo "Unknown arg: $1" >&2; usage ;;
        esac
    done

    local ts run_dir
    ts="$(date +%Y%m%d_%H%M%S)"
    run_dir="${BENCH_ROOT}/${label}_${ts}"
    mkdir -p "$run_dir/logs"

    local manifest="${run_dir}/manifest.tsv"
    echo -e "job_id\tgpus\ttime_min\tsubmit_ts\tsubmit_human\tlabel" > "$manifest"

    # Probe worker — just logs and exits. Heredoc into a file we sbatch.
    local worker="${run_dir}/probe.sh"
    cat > "$worker" <<'EOF'
#!/bin/bash
# Probe: records start timestamp and exits immediately.
set -u
OUT_DIR="${QB_OUT_DIR:?QB_OUT_DIR not set}"
echo "start_ts=$(date +%s)" > "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
echo "start_human=$(date -Iseconds)" >> "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
echo "hostname=$(hostname)" >> "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
exit 0
EOF
    chmod +x "$worker"

    echo "[queue_bench] run_dir = ${run_dir}"
    echo "[queue_bench] times (min) = ${times}"
    echo "[queue_bench] gpus        = ${gpus_list}"
    echo

    for g in $gpus_list; do
        for t in $times; do
            local tstr submit_ts jid
            tstr="$(fmt_time "$t")"
            submit_ts="$(date +%s)"
            jid="$(sbatch --parsable \
                "${SBATCH_COMMON[@]}" \
                --job-name="qb_${label}_g${g}_t${t}" \
                --gpus-per-node="${g}" \
                --time="${tstr}" \
                --output="${run_dir}/logs/g${g}_t${t}_%j.out" \
                --error="${run_dir}/logs/g${g}_t${t}_%j.err" \
                --export="ALL,QB_OUT_DIR=${run_dir}" \
                "$worker")"
            if [[ -z "$jid" ]]; then
                echo "[queue_bench] WARN: sbatch failed for gpus=$g time=$t" >&2
                continue
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$jid" "$g" "$t" "$submit_ts" "$(date -d "@$submit_ts" -Iseconds)" "$label" \
                >> "$manifest"
            echo "  submitted jid=${jid}  gpus=${g}  time=${tstr}"
        done
    done

    echo
    echo "[queue_bench] done submitting. Report when ready with:"
    echo "    $0 report ${run_dir}"
}

cmd_report() {
    local run_dir="${1:-}"
    [[ -n "$run_dir" && -d "$run_dir" ]] || { echo "Usage: $0 report <run-dir>" >&2; exit 1; }
    local manifest="${run_dir}/manifest.tsv"
    [[ -f "$manifest" ]] || { echo "No manifest at $manifest" >&2; exit 1; }

    local csv="${run_dir}/results.csv"
    echo "label,job_id,gpus,time_min,submit_ts,start_ts,wait_sec,state" > "$csv"

    # Pull all job ids, query sacct in one shot
    local ids
    ids="$(awk 'NR>1 {print $1}' "$manifest" | paste -sd, -)"
    local sacct_out
    sacct_out="$(sacct -j "$ids" -n -P -X -o JobID,Start,State 2>/dev/null || true)"

    declare -A START STATE
    while IFS='|' read -r jid start state; do
        [[ -z "$jid" ]] && continue
        START["$jid"]="$start"
        STATE["$jid"]="$state"
    done <<< "$sacct_out"

    local printed=0
    {
        tail -n +2 "$manifest" | while IFS=$'\t' read -r jid gpus tmin submit_ts submit_human label; do
            local start="${START[$jid]:-}"
            local state="${STATE[$jid]:-UNKNOWN}"
            local start_ts="" wait_sec=""
            if [[ -n "$start" && "$start" != "Unknown" && "$start" != "None" ]]; then
                start_ts="$(date -d "$start" +%s 2>/dev/null || true)"
                [[ -n "$start_ts" ]] && wait_sec=$(( start_ts - submit_ts ))
            fi
            echo "$label,$jid,$gpus,$tmin,$submit_ts,${start_ts},${wait_sec},${state}"
        done
    } >> "$csv"

    echo "[queue_bench] wrote ${csv}"
    echo
    printf '%-10s %-12s %-5s %-8s %-12s %-10s\n' "label" "jobid" "gpus" "time_m" "wait_sec" "state"
    tail -n +2 "$csv" | awk -F, '{printf "%-10s %-12s %-5s %-8s %-12s %-10s\n", $1,$2,$3,$4,($7==""?"(pending)":$7),$8}'
}

cmd_list() {
    ls -1dt "${BENCH_ROOT}"/*/ 2>/dev/null || echo "(no runs yet)"
}

cmd_cancel() {
    local run_dir="${1:-}"
    [[ -n "$run_dir" && -f "${run_dir}/manifest.tsv" ]] || { echo "Usage: $0 cancel <run-dir>" >&2; exit 1; }
    local ids
    ids="$(awk 'NR>1 {print $1}' "${run_dir}/manifest.tsv" | tr '\n' ' ')"
    echo "scancel $ids"
    scancel $ids
}

[[ $# -ge 1 ]] || usage
sub="$1"; shift
case "$sub" in
    submit) cmd_submit "$@" ;;
    report) cmd_report "$@" ;;
    list)   cmd_list "$@" ;;
    cancel) cmd_cancel "$@" ;;
    -h|--help) usage ;;
    *) echo "Unknown subcommand: $sub" >&2; usage ;;
esac
