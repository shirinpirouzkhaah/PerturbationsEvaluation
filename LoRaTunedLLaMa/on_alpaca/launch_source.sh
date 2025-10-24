#!/bin/bash

types=(
    "Source"
)

scenarios=("Org" "Mitigation" "Inline" "CoT")

for type in "${types[@]}"; do
    for scenario in "${scenarios[@]}"; do
        sbatch run_generate_source.sh "$type" "$scenario"
    done
done
