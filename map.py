import h5py
import numpy as np
import json
import os
from datetime import datetime

# Definisi elemen target
# versi 35
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

#versi 9
# BASE_ELEMENTS = ["Al", "Fe", "Ca", "Mg"]
# REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

# Konfigurasi
CONFIG = {
    "data": {
        "dataset_path": "20/dataset2k.h5",
        "element_map_path": "element-map-35.json",
        "train_split": "train",
        "val_split": "validation",
        "max_train_samples": 5000,
        "max_val_samples": 2000,
    },
    "results_dir": "r"
}

def load_element_map(element_map_path):
    """Memuat mapping elemen dari file JSON sebagai dictionary."""
    with open(element_map_path, 'r') as f:
        element_map = json.load(f)
    return element_map

def check_dataset(dataset_path, element_map_path, train_split, val_split, max_train_samples, max_val_samples):
    """Memeriksa proporsi atom dalam dataset."""

    # Muat element map sebagai dictionary
    element_map = load_element_map(element_map_path)
    print(f"{datetime.now()} - Element map dimuat: {list(element_map.keys())}")
    print(f"{datetime.now()} - Number of elements/ions in element_map: {len(element_map)}")

    # Verifikasi panjang nilai one-hot
    if not all(len(one_hot) == 35 for one_hot in element_map.values()):
        print(f"{datetime.now()} - Peringatan: Tidak semua nilai one-hot memiliki panjang 35.")
        return

    # Mapping elemen dasar ke indeks kelas berdasarkan one-hot
    element_to_class = {}
    for elem in BASE_ELEMENTS:
        element_to_class[elem] = []
        for class_name, one_hot in element_map.items():
            if class_name.startswith(elem) and class_name in REQUIRED_ELEMENTS:
                idx = np.argmax(one_hot)  # Indeks kelas berdasarkan one-hot
                element_to_class[elem].append(idx)

    # Mapping kelas ke elemen dasar berdasarkan one-hot
    class_to_element = {}
    for class_name, one_hot in element_map.items():
        idx = np.argmax(one_hot)
        elem = next((e for e in BASE_ELEMENTS if class_name.startswith(e)), "background")
        class_to_element[idx] = elem

    # Inisialisasi dictionary untuk menyimpan distribusi
    split_distributions = {"train": {}, "validation": {}}

# Hapus bagian element_counts dan element_proportions
    with h5py.File(dataset_path, 'r') as h5_file:
        for split, max_samples in [(train_split, max_train_samples), (val_split, max_val_samples)]:
            if split not in h5_file:
                print(f"{datetime.now()} - Split '{split}' tidak ditemukan dalam {dataset_path}")
                continue

            group = h5_file[split]
            spectra = group['spectra'][:max_samples]
            labels = group['labels'][:max_samples]
            print(f"{datetime.now()} - Memproses split: {split}, Spectra shape: {spectra.shape}, Labels shape: {labels.shape}")

            # Hitung distribusi kelas di semua posisi
            all_labels = labels.reshape(-1, 35)  # Flattening ke semua posisi
            class_counts = np.sum(all_labels, axis=0)  # Jumlah 1 di setiap kelas
            total_positions = all_labels.shape[0]  # Total posisi (sampel × 4096)

            # Distribusi per kelas
            split_distributions[split] = {}
            for idx, count in enumerate(class_counts):
                class_name = next((name for name, one_hot in element_map.items() if np.argmax(one_hot) == idx), f"Class_{idx}")
                proportion = count / total_positions * 100
                split_distributions[split][class_name] = {"count": int(count), "proportion (%)": round(proportion, 2)}

            # Cetak hasil
            print(f"\n{datetime.now()} - Distribusi Kelas untuk {split} (Total Posisi: {total_positions:,}):")
            for class_name, stats in split_distributions[split].items():
                print(f"  {class_name}: {stats['count']:,} positions ({stats['proportion (%)']:.2f}%)")

            # Simpan hasil ke file
            os.makedirs(CONFIG["results_dir"], exist_ok=True)
            report_path = os.path.join(CONFIG["results_dir"], f"dataset_distribution_{split}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset Distribution Report - {split} (Generated: {datetime.now()})\n")
                f.write(f"Total Positions: {total_positions:,}\n\n")
                f.write("Class Distribution:\n")
                for class_name, stats in split_distributions[split].items():
                    f.write(f"  {class_name}: {stats['count']:,} positions ({stats['proportion (%)']:.2f}%)\n")
            print(f"{datetime.now()} - Laporan distribusi disimpan ke {report_path}")

if __name__ == "__main__":
    check_dataset(
        CONFIG["data"]["dataset_path"],
        CONFIG["data"]["element_map_path"],
        CONFIG["data"]["train_split"],
        CONFIG["data"]["val_split"],
        CONFIG["data"]["max_train_samples"],
        CONFIG["data"]["max_val_samples"]
    )