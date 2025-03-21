#!/bin/bash
#
#SBATCH --job-name=create-images
#SBATCH --partition=compute
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

source ../venv/bin/activate
python create-images.py ../datasets/faceforensics