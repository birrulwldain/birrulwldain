#!/bin/bash
#SBATCH --job-name=spectral_simulation
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/birrulwldain/logs/sim2_%j.out
#SBATCH --error=/home/bwalidain/birrulwldain/logs/sim2_%j.err

# Memuat modul yang diperlukan


# Mengaktifkan lingkungan Conda
source /home/bwalidain/miniconda3/etc/profile.d/conda.sh
conda activate bw

# Mengatur variabel lingkungan untuk optimasi
export MKL_VERBOSE=0  # Untuk debugging MKL
export OMP_NUM_THREADS=8 # Sesuaikan dengan cpus-per-task
export PYTHONPATH=$PYTHONPATH:/home/bwalidain/birrulwldain

# Menjalankan kode Python
python /home/bwalidain/birrulwldain/sim2.py

# Menonaktifkan lingkungan Conda setelah selesai
conda deactivate