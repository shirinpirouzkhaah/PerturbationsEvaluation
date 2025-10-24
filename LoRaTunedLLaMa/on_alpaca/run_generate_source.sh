#!/bin/bash

type="$1"
scenario="$2"
jobname="Lora_${type}_${scenario}"

sbatch --job-name="$jobname" \
       --output="sbatchlogs/${jobname}.out" \
       --error="sbatchlogs/${jobname}.err" \
       <<EOF
#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --constraint=GPUMEM32GB

module load gpu
module load mamba
source activate torchLLR2

python generate_lora_sampling_sbatch_Source.py "$type" "$scenario"
EOF

