#!/bin/bash

#SBATCH -J llamam_test
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -p community
#SBATCH -t infinite

module load cuda/11.8 

source ~/.bashrc

source venv_py310/bin/activate

python attlib/fine_tune.py > debug.txt
