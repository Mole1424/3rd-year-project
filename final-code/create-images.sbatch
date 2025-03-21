#!/bin/bash
#
#SBATCH --job-name=create-images
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

source ../venv/bin/activate
python3.12 create-images.py /dcs/large/u2204489/faceforensics/