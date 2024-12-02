#!/bin/sh
#BSUB -q gpua100
#BSUB -J grid_search_baseline
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 48:00
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source 02456_venv/bin/activate
python3 scripts/grid_search.py > log/grid_search_baseline$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
