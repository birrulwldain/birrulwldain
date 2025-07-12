#!/bin/bash


#SBATCH --job-name=inflibstrain
#SBATCH --partition=short
#SBATCH --nodelist=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00          # Batas waktu 4 jam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=/home/bwalidain/birrulwldain/logs/train-%j.out


source "/home/bwalidain/miniconda3/etc/profile.d/conda.sh"
conda activate rapids-25.06


python "/home/bwalidain/birrulwldain/train.py" \
    --dataset_path "/home/bwalidain/birrulwldain/data/dataset-50.h5" \
    --element_map_path "/home/bwalidain/birrulwldain/data/element-map-18a.json" \
    --model_dir "/home/bwalidain/birrulwldain/models" \
    --results_dir "/home/bwalidain/birrulwldain/logs" \
    --epochs 10 \
    --batch_size 25 \
    --lr 1e-4

