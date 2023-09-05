#!/bin/bash
#SBATCH -N 2  #2 nodes 
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=first_job
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH --partition=gpu
