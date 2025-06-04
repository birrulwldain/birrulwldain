#!/bin/bash
#SBATCH --job-name=spectral_simulation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --output=/home/bwalidain/data/logs/output_spectral_simulation_%j.log
#SBATCH --error=/home/bwalidain/data/logs/error_spectral_simulation_%j.err

# Muat modul
module load anaconda3/2022.9.3

# Aktifkan environment (jika ada)
# conda activate spectral_env  # Uncomment jika menggunakan environment khusus

# Atur variabel lingkungan
export CUDA_VISIBLE_DEVICES=""  # Pastikan tidak ada GPU yang digunakan
export OMP_NUM_THREADS=16       # Sesuaikan dengan cpus-per-task
export MKL_NUM_THREADS=16       # Optimasi untuk Intel MKL

# Pindah ke direktori kerja
cd /home/bwalidain

# Instal dependensi (jalankan sekali, uncomment jika diperlukan)
# pip install --user numpy torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu h5py pandas scipy tqdm

# Buat direktori untuk log dan data jika belum ada
mkdir -p /home/bwalidain/data/logs
mkdir -p /home/bwalidain/data/src
mkdir -p /home/bwalidain/data/out

# Pantau memori sebelum eksekusi
echo "Memori sebelum eksekusi:"
free -h

# Jalankan kode simulasi
echo "Menjalankan spectral_simulation.py..."
python3 /home/bwalidain/spectral_simulation.py

# Periksa status eksekusi
if [ $? -eq 0 ]; then
    echo "Simulasi selesai dengan sukses"
else
    echo "Simulasi gagal, periksa error log di /home/bwalidain/data/logs/error_spectral_simulation_%j.err"
    exit 1
fi

# Pantau memori setelah eksekusi
echo "Memori setelah eksekusi:"
free -h

# Ringkasan log
echo "Log tersimpan di /home/bwalidain/data/logs/output_spectral_simulation_%j.log"
echo "Error (jika ada) tersimpan di /home/bwalidain/data/logs/error_spectral_simulation_%j.err"
echo "File log tambahan tersimpan di /home/bwalidain/data/logs/spectral_simulation_${SLURM_JOB_ID}_*.log"