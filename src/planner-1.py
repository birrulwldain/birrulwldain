# Nama file: planner.py
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm
import logging
import argparse

# --- Konfigurasi ---
# OUTPUT_JSON_PATH sekarang akan dibuat secara dinamis, jadi kita hapus dari sini.
NIST_HDF_PATH = "/home/bwalidain/birrulwldain/data/nist_data(1).h5" # Sesuaikan path ini
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
TEMPERATURE_RANGE = np.linspace(5000, 15000, 100).tolist()
ELECTRON_DENSITY_RANGE = np.logspace(14, 18, 100).tolist()
print("Base elements:", ELECTRON_DENSITY_RANGE)
# --- Setup Logging Sederhana ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def calculate_sampling_weights():
    """Menghitung bobot pemilihan untuk setiap elemen dasar berdasarkan kerapatan garis."""
    logging.info("Membaca data NIST untuk menghitung kerapatan garis...")
    if not os.path.exists(NIST_HDF_PATH):
        raise FileNotFoundError(f"File data NIST tidak ditemukan di: {NIST_HDF_PATH}")
    
    with pd.HDFStore(NIST_HDF_PATH, mode='r') as store:
        df = store.get('nist_spectroscopy_data')
    
    logging.info("Menghitung jumlah garis per elemen dasar...")
    line_counts = {elem: 0 for elem in BASE_ELEMENTS}
    for elem in BASE_ELEMENTS:
        count = len(df[(df['element'] == elem) & (df['sp_num'].isin([1, 2]))])
        line_counts[elem] = count
    
    logging.info("Jumlah garis spektral per elemen dasar:")
    for elem, count in sorted(line_counts.items(), key=lambda item: item[1]):
        logging.info(f"  - {elem}: {count} garis")
        
    sampling_weights = {}
    for elem in BASE_ELEMENTS:
        count = line_counts.get(elem, 0)
        sampling_weights[elem] = 1.0 / np.log10(1 + count) if count > 0 else 1.0

    total_weight = sum(sampling_weights.values())
    normalized_weights = [sampling_weights[elem] / total_weight for elem in BASE_ELEMENTS]
    
    logging.info("Probabilitas pemilihan akhir (setelah normalisasi):")
    for i, elem in enumerate(BASE_ELEMENTS):
        logging.info(f"  - {elem}: {normalized_weights[i]:.4f}")
        
    return normalized_weights

# --- 1. Modifikasi fungsi untuk menerima path output sebagai parameter ---
def generate_combinations_plan(num_samples, output_path):
    """Membuat file JSON berisi rencana pembuatan sampel."""
    logging.info(f"Memulai pembuatan rencana kombinasi untuk {num_samples} sampel...")
    
    sampling_weights = calculate_sampling_weights()
    
    all_combinations = []
    for i in tqdm(range(num_samples), desc="Generating Recipes"):
        temperature = np.random.choice(TEMPERATURE_RANGE)
        electron_density = np.random.choice(ELECTRON_DENSITY_RANGE)
        
        num_target_elements = np.random.randint(15, 17)
        
        selected_elements = np.random.choice(
            BASE_ELEMENTS,
            num_target_elements,
            replace=False,
            p=sampling_weights
        ).tolist()
        
        recipe = {
            "sample_id": i,
            "temperature": temperature,
            "electron_density": electron_density,
            "base_elements": selected_elements
        }
        all_combinations.append(recipe)
        
    logging.info(f"Total {len(all_combinations)} resep berhasil dibuat.")
    
    # Menggunakan path output yang dinamis
    logging.info(f"Menyimpan rencana ke {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_combinations, f, indent=2)
        
    logging.info("Pembuatan rencana selesai.")

# --- 2. Modifikasi blok utama untuk membuat nama file dan memanggil fungsi ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buat rencana kombinasi sampel untuk simulasi spektral.")
    
    parser.add_argument(
        "--num", 
        type=int, 
        default=2000,
        help="Jumlah total sampel kombinasi yang akan dibuat."
    )
    
    args = parser.parse_args()
    
    # Ambil jumlah sampel dari argumen
    num_samples = args.num
    
    # Buat nama file output secara dinamis berdasarkan jumlah sampel
    # Dibagi 1000 sesuai dengan pola yang Anda inginkan (2000 -> 2, 100000 -> 100)
    file_suffix = num_samples // 1000
    output_filename = f"combinations-{file_suffix}.json"
    
    np.random.seed(42)
    
    # Panggil fungsi utama dengan jumlah sampel dan nama file output yang dinamis
    generate_combinations_plan(num_samples, output_filename)