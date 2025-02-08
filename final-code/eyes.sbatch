#!/bin/bash
#
#SBATCH --job-name=eye-landmarking
#SBATCH --partition=gecko,falcon
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA
source ../venv/bin/activate
python3.12 eye_detection.py train