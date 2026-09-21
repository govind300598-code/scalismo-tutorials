#!/bin/bash
# One command: kills anything competing for memory, launches the pipeline, and
# waits by itself -- polling internally -- until it's done or dead. You do not
# need to run tail/ps/free yourself while this is running. Just run this once.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Killing anything that could compete for memory ==="
pkill -9 -f "DensityPipeline" 2>/dev/null
pkill -9 -f "ViewResults" 2>/dev/null
pkill -9 -f "sbt-args" 2>/dev/null
pkill -9 -f "Stage2ReferenceRefinement" 2>/dev/null
sleep 3

echo "=== Launching the pipeline ==="
LOG="pipeline_$(date +%s).log"
setsid nohup bash run_density_pipeline.sh > "$LOG" 2>&1 < /dev/null &
disown
echo "Log file: $LOG"
echo "Waiting -- checking every 30s. This will print ONLY when something changes."
echo "Do not run anything else while this is going."

LAST_LINE=""
while true; do
  sleep 30
  if grep -q "^\[info\] COMPLETE\|^COMPLETE" "$LOG" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "DONE. Pipeline completed successfully."
    echo "=========================================="
    tail -n 25 "$LOG"
    exit 0
  fi
  if grep -qi "OutOfMemoryError\|Exception in thread" "$LOG" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "CRASHED. Here is the error:"
    echo "=========================================="
    grep -i -A 5 "OutOfMemoryError\|Exception in thread" "$LOG" | head -30
    exit 1
  fi
  if ! pgrep -f "sbt-args|DensityPipeline" > /dev/null 2>&1; then
    echo ""
    echo "=========================================="
    echo "STOPPED with no error message (likely killed by the OS). Last output:"
    echo "=========================================="
    tail -n 20 "$LOG"
    exit 1
  fi
  CUR_LINE=$(tail -n 1 "$LOG")
  if [ "$CUR_LINE" != "$LAST_LINE" ]; then
    echo "  ...still running: $CUR_LINE"
    LAST_LINE="$CUR_LINE"
  fi
done
