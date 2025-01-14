#!/bin/bash
#
#SBATCH --job-name=joel-cs310
#SBATCH --partition=falcon,gecko
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00 
#
#SBATCH --mail-type=END,FAIL
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA
source ../venv/bin/activate
python3.12 models.py