#!/usr/bin/env bash
# Launch claim-based workers for one experiment and write the run record.
#   scripts/launch.sh <exp dir> <run id> "<gpu>:<workers> ..." [need_mib]
#   e.g. scripts/launch.sh experiments/E01_three_places E01_20261125_001 "0:2 1:1 2:2 3:1" 3000
# Results: runs/<run id>/results/*.jsonl (one file per job, atomic), logs/worker-g<gpu>-<n>.log, RUN.json.
# Stop gracefully: touch runs/<run id>/STOP (workers finish their current job).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXP=$1; RUN_ID=$2; PLAN=$3; NEED=${4:-3000}
RUN=$ROOT/runs/$RUN_ID
mkdir -p "$RUN/logs"
PY=/opt/conda/bin/python3
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none)
DIRTY=$(git -C "$ROOT" status --porcelain -- src experiments cohorts | wc -l)
REC="{\"run_id\": \"$RUN_ID\", \"exp\": \"$EXP\", \"started_utc\": \"$(date -u +%FT%TZ)\", \"git_commit\": \"$COMMIT\", \"uncommitted_code_files\": $DIRTY, \"workers\": \"$PLAN\", \"need_mib\": $NEED, \"host\": \"$(hostname)\"}"
[ -f "$RUN/RUN.json" ] || echo "$REC" > "$RUN/RUN.json"     # first launch
echo "$REC" >> "$RUN/launches.jsonl"                          # every launch (added workers, restarts)
N0=$(ls "$RUN/logs" | wc -l)
for spec in $PLAN; do
  g=${spec%%:*}; n=${spec##*:}
  for i in $(seq 1 "$n"); do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$ROOT/src \
      nohup "$PY" -m qcbobs.runner work --exp "$EXP" --run "$RUN" --need-mib "$NEED" \
      > "$RUN/logs/worker-g$g-$((N0 + i)).log" 2>&1 &
    echo "gpu $g worker $i pid $!"
    sleep 5    # stagger start-up
  done
done
