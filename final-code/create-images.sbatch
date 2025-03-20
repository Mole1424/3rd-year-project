#!/bin/bash
#
#SBATCH --job-name=create-images
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00

module purge
module load GCCcore/13.3.0 Python/3.12.3
source ../venv/bin/activate

python create-images.py ../datasets/faceforensics