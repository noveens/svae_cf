#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=3-00:00:00
#SBATCH --nodelist=gnode37
#SBATCH --mail-type=NONE

python grid_search.py 2-2
