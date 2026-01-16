#!/bin/bash
# GPU monitoring utility for tracking utilization during training
# Usage: ./monitor-gpu.sh [interval_seconds] [output_file]

set -e

INTERVAL=${1:-2}
OUTPUT=${2:-""}

# Function to get GPU stats
get_gpu_stats() {
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

# Function to get process stats
get_process_stats() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
}

echo "GPU Monitoring Tool"
echo "==================="
echo "Sampling every ${INTERVAL} seconds"
echo "Press Ctrl+C to stop"
echo ""

if [ -n "$OUTPUT" ]; then
    echo "Logging to: $OUTPUT"
    echo "timestamp,gpu_name,pci_bus,driver,pstate,temp_c,gpu_util_%,mem_util_%,mem_total_mb,mem_used_mb,mem_free_mb" > "$OUTPUT"
fi

# Header for display
printf "%-8s %-4s %-6s %-7s %-8s %-8s\n" "Time" "Temp" "GPU%" "Mem%" "MemUsed" "MemTotal"
echo "--------------------------------------------------------"

while true; do
    STATS=$(get_gpu_stats)
    TIMESTAMP=$(date +%H:%M:%S)

    # Parse stats (assuming single GPU for display)
    IFS=',' read -r ts name pci driver pstate temp gpu_util mem_util mem_total mem_used mem_free <<< "$STATS"

    # Display formatted output
    printf "%-8s %-4s %-6s %-7s %-8s %-8s\n" \
        "$TIMESTAMP" \
        "${temp}C" \
        "${gpu_util}%" \
        "${mem_util}%" \
        "${mem_used}MB" \
        "${mem_total}MB"

    # Log to file if specified
    if [ -n "$OUTPUT" ]; then
        echo "$STATS" >> "$OUTPUT"
    fi

    sleep "$INTERVAL"
done
