#!/usr/bin/env bash

# Run probit N times and calculate average throughput

BINARY="${1:-./build/probit}"
RUNS="${2:-20}"
NUM_ELEMENTS="${3:-10000000}"

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found."
    echo "Usage: $0 [binary] [runs] [num_elements]"
    exit 1
fi

echo "Running $BINARY $RUNS times..."
echo

total=0
count=0

for i in $(seq 1 $RUNS); do
    # Run and extract throughput (M/sec line)
    throughput=$($BINARY $NUM_ELEMENTS 2>/dev/null | grep -E "Throughput:" | awk '{print $(NF-1)}')

    if [ ! -z "$throughput" ]; then
        total=$(awk "BEGIN {print $total + $throughput}")
        count=$((count + 1))

        # Progress indicator
        if [ $((i % $RUNS / 10)) -eq 0 ]; then
            echo -n "|"
        fi
    fi
done

echo
echo

if [ $count -gt 0 ]; then
    average=$(awk "BEGIN {printf \"%.2f\", $total / $count}")
    echo "Completed: $count/$RUNS runs"
    echo "Average throughput: $average M/sec"
else
    echo "Error: No valid throughput measurements collected"
    exit 1
fi
