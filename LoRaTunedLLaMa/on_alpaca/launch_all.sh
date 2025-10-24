#!/bin/bash

types=("DataTypeCF" )
scenarios=("Org" "Mitigation" "Inline" "CoT")

for type in "${types[@]}"; do
    for scenario in "${scenarios[@]}"; do
        sbatch run_generate.sh "$type" "$scenario"
    done
done
