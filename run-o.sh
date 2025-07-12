#!/bin/bash


#SBATCH --job-name=inflibstrain
#SBATCH --partition=medium
#SBATCH --nodelist=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00          # Batas waktu 4 jam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=/home/bwalidain/birrulwldain/logs/train-%j.out


echo "========================================================"
echo "Pekerjaan Optimisasi Optuna Dimulai pada: $(date)"
echo "Host: $(hostname)"
echo "GPU yang Dialokasikan: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Mengaktifkan lingkungan Conda
source "/home/bwalidain/miniconda3/etc/profile.d/conda.sh"
conda activate rapids-25.06

#================================================================
# EKSEKUSI SKRIP PYTHON
#================================================================

# REVISI: Menjalankan skrip dalam mode optimisasi hyperparameter.
# --optimize-hparams [JUMLAH]: Menjalankan studi Optuna untuk [JUMLAH] percobaan.
# Contoh di bawah ini akan menjalankan 100 kali percobaan.

echo "Menjalankan skrip Python dalam mode optimisasi Optuna..."
python "/home/bwalidain/birrulwldain/train.py" \
    --dataset_path "/home/bwalidain/birrulwldain/data/dataset-50.h5" \
    --element_map_path "/home/bwalidain/birrulwldain/data/element-map-18a.json" \
    --model_dir "/home/bwalidain/birrulwldain/models" \
    --results_dir "/home/bwalidain/birrulwldain/logs" \
    --optimize-hparams 20

echo "========================================================"
echo "Pekerjaan Optimisasi Optuna Selesai pada: $(date)"
echo "========================================================"
