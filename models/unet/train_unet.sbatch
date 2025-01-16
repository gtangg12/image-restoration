#!/bin/bash

#SBATCH --job-name=ocean     # create a short name for your job
#SBATCH -o /data/vision/beery/scratch/rupa/image-restoration/%j.log
#SBATCH --qos=vision-beery-free
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=96:00:00
#SBATCH --partition=vision-beery

python3 unet.py