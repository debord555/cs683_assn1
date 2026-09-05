#!/bin/bash
# Usage: ./perf_bench.sh <command_to_run> [args...]

CORE=${CPU_CORE:-0}
TMP_LOG=$(mktemp)

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

EVENTS="instructions,l1-dcache-load-misses,l2_cache_misses_from_dc_misses,LLC-load-misses"

echo "Executing 5 runs pinned to CPU core $CORE..."

perf stat -r 5 -x ',' -o "$TMP_LOG" -e "$EVENTS" taskset -c "$CORE" "$@" > /dev/null

awk -F',' '
BEGIN {
    printf "\n%-30s | %18s\n", "Metric", "5-Run Average"
    print "---------------------------------------------------"
}
$1 ~ /^[0-9.]+/ {
    val = $1
    event = $3
    
    if (event ~ /instructions/) name = "Instructions"
    else if (event ~ /l1-dcache-load-misses/) name = "L1D Cache Misses"
    else if (event ~ /CORE_TO_L2/) name = "L2 Cache Misses"
    else if (event ~ /LLC-load-misses/) name = "LLC (L3) Cache Misses"
    else name = event

    printf "%-30s | %18.0f\n", name, val
}
' "$TMP_LOG"

rm -f "$TMP_LOG"