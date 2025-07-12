# =============================================================================
# BAGIAN 1: IMPOR, KONFIGURASI, DAN DEFINISI MODEL
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pandas as pd
import json
import os
import random
from datetime import datetime
from google.colab import drive
from sklearn.metrics import f1_score, accuracy_score, classification_report, hamming_loss
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
from torch.amp import GradScaler, autocast
from collections import defaultdict
import logging
import re
from typing import List, Dict, Tuple, Optional
import sys
from scipy.signal.windows import gaussian
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "colab"
from matplotlib.lines import Line2D
import traceback
import itertools
# Tambahkan impor ini di awal skrip Anda (BAGIAN 1) jika belum ada
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --- Konfigurasi Utama ---
CONFIG = {
    "data": {
        "dataset_path": "/content/diagnostic_validation_set.h5",
        "element_map_path": "/content/element-map-18a.json",
        "field_data_asc_path": "/content/drive/MyDrive/libs_lstm/data/asc/Grup-1.asc",
        "val_split": "validation",
        "max_val_samples": None,
        "subset_size": 1000,
    },
    "model": {
        "input_dim": 1, "d_model": 32, "nhead": 4, "num_encoder_layers": 2,
        "dim_feedforward": 64, "dropout": 0.5, "seq_length": 4096,
        "attn_factor": 5, "num_classes": 18
    },
    "training": {
        "batch_size": 8, "num_epochs": 10, "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_save_path": "/content/drive/MyDrive/libs_lstm/models/informer_multilabel_model_v4.pth",
        "progress_save_path": "/content/drive/MyDrive/libs_lstm/models/training_progress_v4.json",
        "report_save_path": "/content/drive/MyDrive/libs_lstm/results/final_evaluation_report_v4.txt",
        "class_weight_path": "/content/drive/MyDrive/libs_lstm/models/class_weights_multilabel.pth",
        "weight_decay": 1e-3, "gradient_accumulation_steps": 4,
        "early_stopping_patience": 5, "min_delta": 0.001
    }
}

# --- Konfigurasi Simulasi ---
SIMULATION_CONFIG = {
    "resolution": 4096, "wl_range": (200, 900), "sigma": 0.1,
    "target_max_intensity": 0.8, "convolution_sigma": 0.1,
    "temperature_range": np.linspace(5000, 15000, 100).tolist(),
    "electron_density_range": np.logspace(14, 18, 100).tolist(),
    "intensity_threshold": 0.00005,
    "relative_intensity_threshold": 0.01,
    "nist_h5_path": "/content/nist_data(1).h5",
    "atomic_h5_path": "/content/atomic_data1.h5"
}

BASE_ELEMENTS = ["Al", "Ar", "C", "Ca", "Cl", "Co", "Cr", "Fe", "Mg", "Mn", "N", "Na", "Ni", "O", "S", "Si", "Ti"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

# =============================================================================
# BAGIAN 2: DEFINISI KELAS-KELAS INTI
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, attn_factor):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model, self.nhead, self.d_k, self.factor = d_model, nhead, d_model // nhead, attn_factor
        self.q_linear, self.k_linear, self.v_linear, self.out_linear = (nn.Linear(d_model, d_model) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, _ = x.shape; H, D = self.nhead, self.d_k
        Q, K, V = (l(x).view(B, L, H, D).transpose(1, 2) for l in (self.q_linear, self.k_linear, self.v_linear))
        U = min(L, int(self.factor * math.log(L)) if L > 1 else L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None: scores.masked_fill_(mask == 0, -float('inf'))
        top_k, _ = torch.topk(scores, U, dim=-1)
        scores.masked_fill_(scores < top_k[..., -1, None], -float('inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_linear(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attn_factor):
        super(EncoderLayer, self).__init__()
        self.self_attention = ProbSparseSelfAttention(d_model, nhead, dropout, attn_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

class InformerModel(nn.Module):
    def __init__(self, **kwargs):
        super(InformerModel, self).__init__()
        self.d_model = kwargs["d_model"]
        self.embedding = nn.Linear(kwargs["input_dim"], self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, kwargs["seq_length"])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model, nhead=kwargs["nhead"], dim_feedforward=kwargs["dim_feedforward"],
                dropout=kwargs["dropout"], attn_factor=kwargs["attn_factor"]
            ) for _ in range(kwargs["num_encoder_layers"])
        ])
        self.decoder = nn.Linear(self.d_model, kwargs["num_classes"])
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.decoder(x)

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma, self.pos_weight, self.reduction = gamma, pos_weight, reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = ((1 - pt)**self.gamma * bce_loss)
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class HDF5SpectrumDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, split, max_samples=None):
        self.file_path, self.split, self.h5_file = file_path, split, None
        with h5py.File(self.file_path, 'r') as f:
            if split not in f: raise KeyError(f"Split '{split}' tidak ada di {file_path}")
            total_len = len(f[split]['spectra'])
            self.dataset_len = min(total_len, max_samples) if max_samples is not None else total_len
        logging.getLogger('SpectralTraining').info(f"Dataset '{split}' diinisialisasi dengan {self.dataset_len} sampel.")
    def __len__(self): return self.dataset_len
    def __getitem__(self, idx):
        if self.h5_file is None: self.h5_file = h5py.File(self.file_path, 'r')
        spectra = self.h5_file[self.split]['spectra'][idx][..., np.newaxis].astype(np.float32)
        labels = self.h5_file[self.split]['labels'][idx].astype(np.float32)
        return torch.from_numpy(spectra), torch.from_numpy(labels)

class DataFetcher:
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path; self.logger = logging.getLogger('SpectralTraining')
    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store: df = store.get('nist_spectroscopy_data')
            filtered_df = df[(df['element'] == element) & (df['sp_num'] == int(sp_num))].copy()
            required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
            if filtered_df.empty or not all(col in filtered_df.columns for col in required_columns): return [], 0.0

            # PERBAIKAN: Gunakan .loc untuk menghindari SettingWithCopyWarning
            for col in required_columns:
                filtered_df.loc[:, col] = pd.to_numeric(filtered_df[col], errors='coerce')

            filtered_df = filtered_df.dropna(subset=required_columns)
            filtered_df = filtered_df[(filtered_df['ritz_wl_air(nm)'] >= SIMULATION_CONFIG["wl_range"][0]) & (filtered_df['ritz_wl_air(nm)'] <= SIMULATION_CONFIG["wl_range"][1])]
            if filtered_df.empty: return [], 0.0
            return filtered_df[required_columns].values.tolist(), 0.0
        except Exception as e:
            self.logger.error(f"Error mengambil data NIST untuk {element}_{sp_num}: {e}"); return [], 0.0

class SpectrumSimulator:
    def __init__(self, nist_data, element, ion, ionization_energy, config, element_map_labels):
        self.nist_data, self.element, self.ion, self.ionization_energy = nist_data, element, ion, ionization_energy
        self.resolution, self.wl_range, self.sigma = config["resolution"], config["wl_range"], config["sigma"]
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.element_label = f"{element}_{ion}"
    def _partition_function(self, energy_levels, degeneracies, temperature):
        k_B = 8.617333262145e-5
        return sum(g * np.exp(-E / (k_B * temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0
    def _calculate_intensity(self, temperature, energy, degeneracy, einstein_coeff, Z):
        k_B = 8.617333262145e-5
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * temperature))) / Z
    def simulate_single_temp(self, temp: float, atom_percentage: float = 1.0):
        if not self.nist_data: return None
        levels = {}
        for _, _, Ek, Ei, gi, gk in self.nist_data:
            if all(v is not None for v in [Ek, Ei, gi, gk]):
                levels[float(Ei)], levels[float(Ek)] = float(gi), float(gk)
        if not levels: return None
        Z = self._partition_function(list(levels.keys()), list(levels.values()), temp)
        intensities = np.zeros(self.resolution, dtype=np.float32)
        for wl, Aki, Ek, _, _, gk in self.nist_data:
            intensity = self._calculate_intensity(temp, Ek, gk, Aki, Z)
            idx = np.searchsorted(self.wavelengths, wl)
            if 0 <= idx < self.resolution: intensities[idx] += intensity * atom_percentage
        return intensities, intensities.copy()

class MixedSpectrumSimulator:
    def __init__(self, simulators, config, element_map_labels):
        self.simulators, self.config, self.element_map_labels = simulators, config, element_map_labels
        self.resolution, self.wl_range = config["resolution"], config["wl_range"]
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.num_labels = len(next(iter(element_map_labels.values())))
    def _normalize_intensity(self, intensity, target_max):
        max_val = np.max(intensity); return (intensity / max_val * target_max).astype(np.float32) if max_val > 0 else intensity
    def _convolve_spectrum(self, spectrum, sigma_nm):
        wl_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = sigma_nm / wl_step
        kernel = gaussian(int(6 * sigma_points) | 1, sigma_points); kernel /= np.sum(kernel)
        return np.convolve(spectrum, kernel, mode='same').astype(np.float32)
    def _saha_ratio(self, ion_energy, temp, electron_density):
        k_B, m_e, h = 8.617333262e-5, 9.1093837e-31, 4.135667696e-15
        kT_eV = k_B * temp
        saha_factor = 2 * ((2 * np.pi * m_e * kT_eV * 1.60218e-19) / (h * 1.60218e-19)**2)**1.5
        return (saha_factor / (electron_density * 1e6)) * np.exp(-ion_energy / kT_eV)

    def generate_sample(self, ionization_energies: Dict[str, float], selected_base_elements: List[str], temp: float, electron_density: float):
        # Logika konsentrasi acak dari fungsi awal Anda
        atom_percentages_dict = {}
        total_target_percentage = 0.0
        for base_elem in selected_base_elements:
            elem_neutral = f"{base_elem}_1"
            elem_ion = f"{base_elem}_2"
            ion_energy = ionization_energies.get(f"{base_elem} I", 7.0)
            saha_ratio = self._saha_ratio(ion_energy, temp, electron_density)
            # Memberikan konsentrasi total acak untuk setiap elemen
            total_percentage = np.random.uniform(0.1, 5.0) 
            atom_percentages_dict[elem_neutral] = total_percentage * (1 / (1 + saha_ratio))
            atom_percentages_dict[elem_ion] = total_percentage * (saha_ratio / (1 + saha_ratio))
            total_target_percentage += total_percentage
        
        if total_target_percentage > 0:
            scaling_factor = 100.0 / total_target_percentage
            for key in atom_percentages_dict:
                atom_percentages_dict[key] *= scaling_factor

        selected_simulators = [sim for sim in self.simulators if sim.element_label in atom_percentages_dict]
        
        if not selected_simulators:
            normalized_spectrum = np.random.normal(0, 0.01, self.resolution).astype(np.float32)
            labels = np.zeros((self.resolution, self.num_labels), dtype=np.float32)
            background_label = np.array(self.element_map_labels["background"], dtype=np.float32)
            labels[:] = background_label
            return normalized_spectrum, labels

        mixed_spectrum = np.zeros(self.resolution, dtype=np.float32)
        element_contributions = np.zeros((len(selected_simulators), self.resolution), dtype=np.float32)

        for sim_idx, simulator in enumerate(selected_simulators):
            atom_percentage = atom_percentages_dict.get(simulator.element_label, 0.0)
            if atom_percentage > 0:
                result = simulator.simulate_single_temp(temp, atom_percentage)
                if result is not None:
                    spectrum, contrib = result
                    mixed_spectrum += spectrum
                    element_contributions[sim_idx] = contrib

        if np.max(mixed_spectrum) == 0:
            return None

        convolved_spectrum = self._convolve_spectrum(mixed_spectrum, self.config["convolution_sigma"])
        normalized_spectrum = self._normalize_intensity(convolved_spectrum, self.config["target_max_intensity"])

        # Pembuatan label menggunakan threshold absolut
        active_mask = element_contributions >= self.intensity_threshold
        
        active_labels_matrix = np.array(
            [self.element_map_labels.get(sim.element_label) for sim in selected_simulators], 
            dtype=np.float32
        )
        
        # Perbaikan untuk menangani edge case dimensi
        active_labels_matrix = np.atleast_2d(active_labels_matrix)

        labels = (active_mask.T @ active_labels_matrix > 0).astype(np.float32)
        
        is_active = labels.sum(axis=1) > 0
        background_label = np.array(self.element_map_labels["background"], dtype=np.float32)
        labels[~is_active] = background_label
        
        return normalized_spectrum, labels

class IntegratedSimulatedDataset(torch.utils.data.Dataset):
    def __init__(self, recipes, simulators, ionization_energies, element_map, sim_config):
        self.recipes, self.simulators, self.ionization_energies, self.element_map, self.sim_config = recipes, simulators, ionization_energies, element_map, sim_config
        self.mixed_simulator = MixedSpectrumSimulator(self.simulators, self.sim_config, self.element_map)
        logging.getLogger('SpectralTraining').info(f"Dataset terintegrasi dibuat dengan {len(self.recipes)} resep.")
    def __len__(self): return len(self.recipes)
    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        result = self.mixed_simulator.generate_sample(self.ionization_energies, recipe["base_elements"], recipe["temperature"], recipe["electron_density"])
        if result is None: return self.__getitem__(random.randint(0, len(self) - 1))
        spectrum_np, labels_np = result
        spectrum_np += np.random.normal(0, 0.005, spectrum_np.shape)
        return torch.from_numpy(spectrum_np[..., np.newaxis].copy()), torch.from_numpy(labels_np.copy())

def calculate_sampling_weights(sim_config, base_elements):
    logger = logging.getLogger('SpectralTraining')
    logger.info("Membaca data NIST untuk estimasi...")
    nist_h5_path, wl_range, resolution, intensity_threshold = (sim_config[k] for k in ["nist_h5_path", "wl_range", "resolution", "intensity_threshold"])
    if not os.path.exists(nist_h5_path): raise FileNotFoundError(f"File NIST tidak ditemukan: {nist_h5_path}")
    with pd.HDFStore(nist_h5_path, mode='r') as store: df = store.get('nist_spectroscopy_data')
    wavelengths = np.linspace(wl_range[0], wl_range[1], resolution, dtype=np.float32)
    estimated_active_positions_per_ion = defaultdict(float)
    filtered_df = df[df['sp_num'].isin([1, 2])]
    for elem in tqdm(base_elements, desc="Estimasi Efisiensi Elemen"):
        for sp_num in [1, 2]:
            ion_key = f"{elem}_{sp_num}"
            elem_ion_df = filtered_df[(filtered_df['element'] == elem) & (filtered_df['sp_num'] == sp_num)]
            ion_spectrum_temp = np.zeros(resolution, dtype=np.float32)
            for _, row in elem_ion_df.iterrows():
                wl = pd.to_numeric(row.get('ritz_wl_air(nm)'), errors='coerce')
                aki = pd.to_numeric(row.get('Aki(s^-1)'), errors='coerce')
                if pd.notna(wl) and pd.notna(aki) and wl_range[0] <= wl <= wl_range[1]:
                    ion_spectrum_temp[np.searchsorted(wavelengths, wl)] += float(aki)
            active_positions_count = np.sum(ion_spectrum_temp > intensity_threshold)
            estimated_active_positions_per_ion[ion_key] = float(active_positions_count)
    estimated_active_positions_per_element_total = {elem: max(estimated_active_positions_per_ion.get(f"{elem}_1", 0.0) + estimated_active_positions_per_ion.get(f"{elem}_2", 0.0), 1.0) for elem in base_elements}
    logger.info("Estimasi posisi label aktif selesai.")
    return estimated_active_positions_per_element_total


    logger = logging.getLogger('SpectralTraining')
    logger.info(f"Membuat rencana kombinasi untuk {num_samples} sampel...")
    active_positions = calculate_sampling_weights(sim_config, base_elements)
    target_positions_per_element = (num_samples * sim_config["resolution"] * 0.8) / len(base_elements)
    appearances_per_element = {e: math.ceil(target_positions_per_element / pos) for e, pos in active_positions.items()}
    total_appearances_needed = int(num_samples * 13.5)
    total_appearances_current = sum(appearances_per_element.values())
    if total_appearances_current > 0:
        scaling_factor = total_appearances_needed / total_appearances_current
        appearances_per_element = {e: max(1, math.floor(n * scaling_factor)) for e, n in appearances_per_element.items()}
    all_combinations, remaining_appearances = [], appearances_per_element.copy()
    for sample_id in tqdm(range(num_samples), desc="Generating Recipes"):
        num_elements = np.random.randint(10, 18)
        available_elements = [e for e in base_elements if remaining_appearances.get(e, 0) > 0]
        if len(available_elements) < num_elements:
            remaining_appearances = appearances_per_element.copy()
            available_elements = base_elements
        available_elements.sort(key=lambda e: remaining_appearances[e], reverse=True)
        selected_elements = available_elements[:num_elements]
        for e in selected_elements: remaining_appearances[e] -= 1
        recipe = {"sample_id": sample_id, "temperature": float(np.random.choice(sim_config["temperature_range"])), "electron_density": float(np.random.choice(sim_config["electron_density_range"])), "base_elements": selected_elements}
        all_combinations.append(recipe)
    logger.info(f"Total {len(all_combinations)} resep dibuat. Menyimpan ke {output_path}...")
    with open(output_path, 'w') as f: json.dump(all_combinations, f, indent=2)
    return all_combinations

# =============================================================================
# BAGIAN 3: FUNGSI-FUNGSI HELPER UNTUK ANALISIS
# =============================================================================
def run_final_evaluation(model, element_map, config, dataset_to_eval, dataset_name: str, threshold: float = 0.5):
    logger = logging.getLogger('SpectralTraining')
    logger.info(f"\n--- MEMULAI EVALUASI KUANTITATIF FINAL PADA SET '{dataset_name.upper()}' ---")
    device = next(model.parameters()).device
    num_classes = config["model"]["num_classes"]
    dl_args = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}
    test_loader = DataLoader(dataset_to_eval, batch_size=16, **dl_args)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc=f"Mengevaluasi pada set {dataset_name}"):
            data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(data)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > threshold).int()
            all_preds.append(predictions.cpu())
            all_targets.append(target.cpu())

    y_pred = torch.cat(all_preds).view(-1, num_classes).numpy()
    y_true = torch.cat(all_targets).view(-1, num_classes).numpy()

    logger.info("\nMenghitung metrik performa...")
    class_names = list(element_map.keys())
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    class_report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4)

    final_metrics = {
        "Subset Accuracy (Exact Match)": accuracy_score(y_true, y_pred), "Hamming Loss": hamming_loss(y_true, y_pred),
        "F1-Score (Micro Average)": report_dict['micro avg']['f1-score'], "F1-Score (Macro Average)": report_dict['macro avg']['f1-score'],
        "F1-Score (Samples Average)": f1_score(y_true, y_pred, average='samples', zero_division=0),
        "Precision (Macro Average)": report_dict['macro avg']['precision'], "Recall (Macro Average)": report_dict['macro avg']['recall']
    }

    report_content = f"--- HASIL EVALUASI FINAL PADA '{dataset_name.upper()}' ---\n\n"
    for name, value in final_metrics.items():
        report_content += f"{name:<35}: {value:.4f}\n"
    report_content += f"\n--- Laporan Performa per Elemen (Kelas) ---\n{class_report_str}"

    print("\n" + report_content)
    report_path = config["training"]["report_save_path"]
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"\nLaporan berhasil diekspor ke: {report_path}")

def create_static_plot(
    wavelengths: np.ndarray, 
    spectrum: np.ndarray, 
    predictions: np.ndarray,
    class_names: list, 
    ground_truth: np.ndarray = None,
    title: str = "Visualisasi Prediksi Elemen (Statis)"
    ):
    """
    Membuat plot statis menggunakan Matplotlib untuk debug.
    Menampilkan spektrum, ground truth, dan prediksi.
    """
    try:
        bg_index = class_names.index("background")
        class_names_no_bg = [name for i, name in enumerate(class_names) if i != bg_index]
        if ground_truth is not None:
            ground_truth_no_bg = np.delete(ground_truth, bg_index, axis=1)
        predictions_no_bg = np.delete(predictions, bg_index, axis=1)
    except ValueError:
        class_names_no_bg = class_names
        ground_truth_no_bg = ground_truth
        predictions_no_bg = predictions

    num_classes_no_bg = len(class_names_no_bg)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    
    axes[0].plot(wavelengths, spectrum, color='black', linewidth=1)
    axes[0].set_title(title, fontsize=16)
    axes[0].set_ylabel("Intensitas")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    if ground_truth is not None:
        axes[1].imshow(ground_truth_no_bg.T, aspect='auto', interpolation='nearest',
                       cmap=mcolors.ListedColormap(['white', 'cornflowerblue']),
                       extent=[wavelengths.min(), wavelengths.max(), -0.5, num_classes_no_bg - 0.5])
        axes[1].set_yticks(range(num_classes_no_bg))
        axes[1].set_yticklabels(class_names_no_bg, fontsize=8)
        axes[1].set_ylabel("Ground Truth")

    axes[2].imshow(predictions_no_bg.T, aspect='auto', interpolation='nearest',
                   cmap=mcolors.ListedColormap(['white', 'salmon']),
                   extent=[wavelengths.min(), wavelengths.max(), -0.5, num_classes_no_bg - 0.5])
    axes[2].set_yticks(range(num_classes_no_bg))
    axes[2].set_yticklabels(class_names_no_bg, fontsize=8)
    axes[2].set_ylabel("Prediksi")
    axes[2].set_xlabel("Panjang Gelombang (nm)")

    plt.tight_layout(h_pad=0.5)
    plt.show()

def analyze_from_recipe(model, element_map, config, sim_factory, recipe, peak_height, threshold):
    """
    Menganalisis satu sampel yang dihasilkan dari 'resep' untuk visualisasi.
    """
    logger = logging.getLogger('SpectralTraining')
    sample_id = recipe.get('sample_id', 'N/A')
    print(f"\n--- Menganalisis Sampel dari Resep #{sample_id} ---")

    result = sim_factory.mixed_simulator.generate_sample(
        sim_factory.ionization_energies,
        recipe["base_elements"],
        recipe["temperature"],
        recipe["electron_density"]
    )
    if result is None:
        logger.error(f"Gagal menghasilkan spektrum untuk resep #{sample_id}.")
        return

    spectrum, ground_truth = result
    wavelengths = sim_factory.mixed_simulator.wavelengths 
    class_names = list(element_map.keys())
    device = next(model.parameters()).device
    
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)
    
    create_static_plot(
        wavelengths=wavelengths,
        spectrum=spectrum,
        predictions=predictions,
        class_names=class_names,
        ground_truth=ground_truth,
        title=f"Analisis Statis Resep #{sample_id}"
    )

def analyze_from_asc(model, element_map, config, asc_file_path, peak_height, threshold):
    print(f"\n--- Menganalisis Data Lapangan dari File: {os.path.basename(asc_file_path)} ---")
    def _prepare_asc_file(asc_path, h5_grid_path):
        df = pd.read_csv(asc_path, sep=r'\s+', names=['wavelength', 'intensity'], comment='#')
        original_wl, original_spec = df['wavelength'].values, df['intensity'].values
        # Menggunakan path dari CONFIG untuk konsistensi
        with h5py.File(config["data"]["dataset_path"], 'r') as f: target_wl = f['wavelengths'][:]
        baseline = np.percentile(original_spec, 5)
        spec_corrected = original_spec - baseline; spec_corrected[spec_corrected < 0] = 0
        max_val = np.max(spec_corrected)
        spec_normalized = (spec_corrected / max_val) * 0.8 if max_val > 0 else spec_corrected
        return target_wl, np.interp(target_wl, original_wl, spec_normalized).astype(np.float32)
    try:
        wavelengths, spectrum = _prepare_asc_file(asc_file_path, config["data"]["dataset_path"])
    except Exception as e: print(f"Gagal memproses file .asc: {e}"); return
    class_names, device = list(element_map.keys()), next(model.parameters()).device
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)
    # Gunakan plot statis juga untuk data lapangan agar konsisten
    create_static_plot(
        wavelengths=wavelengths, 
        spectrum=spectrum, 
        predictions=predictions, 
        class_names=class_names, 
        ground_truth=None, # Tidak ada ground truth untuk data lapangan
        title=f"Prediksi pada Data Lapangan ({os.path.basename(asc_file_path)})"
    )

def setup_logging():
    logger = logging.getLogger('SpectralTraining')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def set_seed(seed_value=42):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train_one_epoch(model, data_loader, criterion, optimizer, device, grad_steps, scheduler=None):
    model.train(); total_loss = 0.0; scaler = GradScaler(); optimizer.zero_grad()
    pbar = tqdm(data_loader, desc="Training Epoch", leave=False)
    for i, (data, labels) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type=str(device), dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target) / grad_steps
        scaler.scale(loss).backward()
        if (i + 1) % grad_steps == 0 or (i + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
        total_loss += loss.item() * grad_steps
        pbar.set_postfix({'Loss': f'{total_loss / (i+1):.4f}'})
    return total_loss / len(data_loader)

def validate_one_epoch(model, data_loader, criterion, device, num_classes):
    model.eval(); val_loss = 0.0; all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Validation Epoch", leave=False):
            data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=str(device), dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).int()
            all_preds.append(preds.cpu()); all_targets.append(target.cpu())
    y_pred = torch.cat(all_preds).view(-1, num_classes).numpy()
    y_true = torch.cat(all_targets).view(-1, num_classes).numpy()
    val_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    val_acc = accuracy_score(y_true, y_pred)
    return val_loss / len(data_loader), val_f1, val_acc

def plot_and_save_history(history, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    train_loss_hist, val_loss_hist, val_f1_hist, val_acc_hist = (history.get(k, []) for k in ['train_loss', 'val_loss', 'val_f1', 'val_acc'])
    if not train_loss_hist: print("Tidak ada data histori untuk di-plot."); return
    epochs = range(1, len(train_loss_hist) + 1)
    ax1.plot(epochs, train_loss_hist, 'bo-', label='Training Loss'); ax1.plot(epochs, val_loss_hist, 'ro-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, linestyle=':')
    ax2.plot(epochs, val_f1_hist, 'go-', label='Validation F1-Score (Samples)'); ax2.set_title('Validation F1-Score')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1-Score'); ax2.grid(True, linestyle=':')
    ax2b = ax2.twinx(); ax2b.plot(epochs, val_acc_hist, 'ms--', label='Validation Accuracy (Subset)', alpha=0.6)
    ax2b.set_ylabel('Accuracy (Subset)', color='m'); ax2b.tick_params(axis='y', labelcolor='m')
    lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.tight_layout(); save_path = os.path.join(save_dir, "training_history_plot.png"); plt.savefig(save_path)
    print(f"\nPlot histori training disimpan di: {save_path}"); plt.show()

def generate_diagnostic_recipes(num_pure=10, num_pairs=100, num_trios=150, num_complex=50):
    """
    REVISI: Loop untuk membuat resep "noise" telah dihapus.
    """
    logger = logging.getLogger('SpectralTraining')
    recipes = []
    sample_id_counter = 0
    logger.info(f"Membuat resep diagnostik (tanpa noise)...")

    # Kasus elemen murni
    for elem in BASE_ELEMENTS:
        for _ in range(num_pure):
            recipes.append({
                "sample_id": sample_id_counter, 
                "category": "pure", 
                "temperature": float(np.random.choice(SIMULATION_CONFIG["temperature_range"])), 
                "electron_density": float(np.random.choice(SIMULATION_CONFIG["electron_density_range"])), 
                "base_elements": [elem]
            })
            sample_id_counter += 1

    # Kasus campuran 2 elemen
    element_pairs = list(itertools.combinations(BASE_ELEMENTS, 2))
    for _ in range(num_pairs):
        recipes.append({
            "sample_id": sample_id_counter, 
            "category": "pair", 
            "temperature": float(np.random.choice(SIMULATION_CONFIG["temperature_range"])), 
            "electron_density": float(np.random.choice(SIMULATION_CONFIG["electron_density_range"])), 
            "base_elements": list(random.choice(element_pairs))
        })
        sample_id_counter += 1

    # Kasus campuran 3 elemen
    element_trios = list(itertools.combinations(BASE_ELEMENTS, 3))
    for _ in range(num_trios):
        recipes.append({
            "sample_id": sample_id_counter, 
            "category": "trio", 
            "temperature": float(np.random.choice(SIMULATION_CONFIG["temperature_range"])), 
            "electron_density": float(np.random.choice(SIMULATION_CONFIG["electron_density_range"])), 
            "base_elements": list(random.choice(element_trios))
        })
        sample_id_counter += 1
        
    # Kasus campuran kompleks
    for _ in range(num_complex):
        num_elems_complex = np.random.randint(10, 18)
        recipes.append({
            "sample_id": sample_id_counter, 
            "category": "complex", 
            "temperature": float(np.random.choice(SIMULATION_CONFIG["temperature_range"])), 
            "electron_density": float(np.random.choice(SIMULATION_CONFIG["electron_density_range"])), 
            "base_elements": random.sample(BASE_ELEMENTS, num_elems_complex)
        })
        sample_id_counter += 1

    logger.info(f"Total {len(recipes)} resep diagnostik berhasil dibuat.")
    return recipes

def generate_combinations_plan(num_samples: int, sim_config: dict, base_elements: list, exclude_signatures: set):
    logger = logging.getLogger('SpectralTraining')
    logger.info(f"Membuat {num_samples} resep training unik (menghindari {len(exclude_signatures)} resep terlarang)...")
    all_combinations = []
    pbar = tqdm(total=num_samples, desc="Generating Unique Training Recipes")
    while len(all_combinations) < num_samples:
        selected_elements = sorted(random.sample(base_elements, np.random.randint(10, 18)))
        temperature = float(np.random.choice(sim_config["temperature_range"]))
        electron_density = float(np.random.choice(sim_config["electron_density_range"]))
        signature = (tuple(selected_elements), temperature, electron_density)
        if signature in exclude_signatures: continue
        recipe = {"sample_id": len(all_combinations), "temperature": temperature, "electron_density": electron_density, "base_elements": list(selected_elements)}
        all_combinations.append(recipe)
        pbar.update(1)
    pbar.close()
    return all_combinations


def plot_random_sample_from_loader(model, data_loader, sim_factory, element_map, device, title):
    """
    Mengambil satu batch dari DataLoader, memilih sampel pertama, dan mem-plotnya
    menggunakan create_static_plot untuk tujuan debug.
    """
    if not data_loader:
        print(f"Peringatan: DataLoader untuk '{title}' kosong, plot dilewati.")
        return

    # Ambil satu batch data secara acak
    try:
        spectra_batch, labels_batch = next(iter(data_loader))
    except StopIteration:
        print(f"Peringatan: DataLoader untuk '{title}' tidak bisa menghasilkan data, plot dilewati.")
        return

    # Pilih sampel pertama dari batch
    spectrum_tensor = spectra_batch[0]
    ground_truth = labels_batch[0].cpu().numpy()

    # Siapkan input untuk model
    input_tensor = spectrum_tensor.unsqueeze(0).to(device)
    
    # Lakukan prediksi
    model.eval() # Pastikan model dalam mode evaluasi untuk plotting
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > 0.7).astype(int)
    model.train() # Kembalikan model ke mode training setelahnya

    # Siapkan data untuk plotting
    wavelengths = sim_factory.mixed_simulator.wavelengths
    spectrum_np = spectrum_tensor.squeeze().cpu().numpy()
    class_names = list(element_map.keys())

    # Panggil fungsi plot statis
    create_static_plot(
        wavelengths=wavelengths,
        spectrum=spectrum_np,
        predictions=predictions,
        class_names=class_names,
        ground_truth=ground_truth,
        title=title
    )

# =============================================================================
# BAGIAN 4: KELAS-KELAS PIPELINE
# =============================================================================

class SimulationFactory:
    def __init__(self, sim_config, base_elements, element_map_path):
        self.logger = logging.getLogger('SpectralTraining')
        self.logger.info("--- FASE INISIALISASI PABRIK SIMULASI ---")
        self.sim_config = sim_config
        self.base_elements = base_elements
        with open(element_map_path, 'r') as f: self.element_map = json.load(f)

        # Memuat energi ionisasi
        with pd.HDFStore(sim_config["atomic_h5_path"], 'r') as store:
            df_ionization = store.get('elements')
        self.ionization_energies = {row['Sp. Name']: float(row['Ionization Energy (eV)']) for _, row in df_ionization.iterrows()}

        self.logger.info("Mengambil data spektral dari NIST...")
        fetcher = DataFetcher(sim_config["nist_h5_path"])
        nist_data_dict = {elem: fetcher.get_nist_data(*elem.split('_'))[0] for elem in tqdm(REQUIRED_ELEMENTS, desc="Fetching NIST Data")}

        self.logger.info("Mempersiapkan 'pabrik' simulator untuk setiap ion...")
        self.base_simulators = []
        for elem_key, nist_data in nist_data_dict.items():
            if nist_data:
                element, ion_str = elem_key.split('_')
                ion_name_suffix = 'I' if ion_str == '1' else 'II'
                ion_energy = self.ionization_energies.get(f"{element} {ion_name_suffix}", 7.0)
                self.base_simulators.append(SpectrumSimulator(nist_data, element, int(ion_str), ion_energy, self.sim_config, self.element_map))
        self.logger.info(f"Total {len(self.base_simulators)} simulator dasar berhasil dibuat.")

        ### PERBAIKAN ###
        # Inisialisasi MixedSimulator di sini agar bisa diakses dari luar.
        self.mixed_simulator = MixedSpectrumSimulator(self.base_simulators, self.sim_config, self.element_map)

    def create_dataset_from_recipes(self, recipes):
        return IntegratedSimulatedDataset(
            recipes=recipes,
            simulators=self.base_simulators,
            ionization_energies=self.ionization_energies,
            element_map=self.element_map,
            sim_config=self.sim_config
        )
class TrainingPipeline:
    def __init__(self, config, sim_factory, validation_recipes, reserved_signatures):
        self.config, self.training_config, self.sim_factory = config, config['training'], sim_factory
        self.validation_recipes = validation_recipes
        self.reserved_signatures = reserved_signatures
        self.logger = logging.getLogger('SpectralTraining')
        self.device = self.training_config['device']

        self.model = InformerModel(**config['model']).to(self.device)
        self._load_progress_and_weights()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.training_config['learning_rate'], weight_decay=self.training_config['weight_decay'])

        if int(torch.__version__.split('.')[0]) >= 2:
            self.logger.info("Mengompilasi model dengan torch.compile()...")
            self.model = torch.compile(self.model)

        class_weights = torch.load(self.training_config['class_weight_path']).to(self.device)
        dampened_weights = 1.0 + torch.log(class_weights)
        self.criterion = MultiLabelFocalLoss(pos_weight=dampened_weights, gamma=2.0)
        self.scheduler = None

    def _load_progress_and_weights(self):
        self.start_epoch, self.best_val_f1 = 1, 0.0
        self.histories = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
        progress_path, model_path = self.training_config['progress_save_path'], self.training_config['model_save_path']
        if os.path.exists(progress_path):
            self.logger.info(f"Memuat progres dari {progress_path}")
            with open(progress_path, 'r') as f: progress_data = json.load(f)
            metadata = progress_data.get("metadata", {})
            self.best_val_f1 = metadata.get("best_f1_score", 0.0)
            self.start_epoch = metadata.get("last_saved_epoch", 0) + 1
            self.histories = progress_data.get("histories", self.histories)
            self.logger.info(f"Melanjutkan dari Epoch {self.start_epoch} dengan F1 terbaik: {self.best_val_f1:.4f}")
        if os.path.exists(model_path):
            self.logger.info(f"Memuat bobot model dari {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean_state_dict, strict=False)
            self.logger.info("Bobot model berhasil dimuat.")

    def run(self):
        self.logger.info("--- MEMULAI PROSES TRAINING ---")
        epochs_no_improve = 0
        dl_args = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}

        val_dataset = self.sim_factory.create_dataset_from_recipes(self.validation_recipes)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config['batch_size'] * 2, shuffle=False, **dl_args)

        for epoch in range(self.start_epoch, self.training_config['num_epochs'] + 1):
            self.logger.info(f"--- Memulai Epoch {epoch}/{self.training_config['num_epochs']} ---")

            train_recipes = generate_combinations_plan(
                self.config['data']['subset_size'], self.sim_factory.sim_config,
                self.sim_factory.base_elements, self.reserved_signatures
            )
            train_dataset = self.sim_factory.create_dataset_from_recipes(train_recipes)
            train_loader = DataLoader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True, **dl_args)

            if self.scheduler is None:
                self.logger.info("Menginisialisasi scheduler OneCycleLR...")
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.training_config['learning_rate'], epochs=self.training_config['num_epochs'], steps_per_epoch=len(train_loader), pct_start=0.3)

            train_loss = train_one_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device, self.training_config['gradient_accumulation_steps'], self.scheduler)
            val_loss, val_f1, val_acc = validate_one_epoch(self.model, val_loader, self.criterion, self.device, self.config['model']['num_classes'])

            self._log_and_save(epoch, train_loss, val_loss, val_f1, val_acc)
            self.logger.info(f"Membuat plot debug untuk Epoch {epoch}...")
            
            # Plot satu sampel acak dari set training
            plot_random_sample_from_loader(
                model=self.model,
                data_loader=train_loader,
                sim_factory=self.sim_factory,
                element_map=self.sim_factory.element_map,
                device=self.device,
                title=f"Contoh Sampel Training - Epoch {epoch}"
            )

            # Plot satu sampel acak dari set validasi
            plot_random_sample_from_loader(
                model=self.model,
                data_loader=val_loader,
                sim_factory=self.sim_factory,
                element_map=self.sim_factory.element_map,
                device=self.device,
                title=f"Contoh Sampel Validasi - Epoch {epoch}"
            )
            if val_f1 > self.best_val_f1 + self.training_config['min_delta']:
                self.best_val_f1 = val_f1; epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.training_config['early_stopping_patience']:
                self.logger.info("Batas kesabaran Early Stopping tercapai."); break

        self.logger.info("--- PELATIHAN SELESAI ---")
        if self.histories['train_loss']: plot_and_save_history(self.histories, os.path.dirname(self.training_config['report_save_path']))
        return self.model, self.sim_factory.element_map

    def _log_and_save(self, epoch, train_loss, val_loss, val_f1, val_acc):
        self.histories['train_loss'].append(train_loss); self.histories['val_loss'].append(val_loss); self.histories['val_f1'].append(val_f1); self.histories['val_acc'].append(val_acc)
        self.logger.info(f"Epoch {epoch} Selesai | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        action_taken = ""
        if val_f1 > self.best_val_f1 + self.training_config['min_delta']:
            action_taken = f"F1 meningkat signifikan ke {val_f1:.4f}. Model disimpan."
            torch.save(self.model.state_dict(), self.training_config['model_save_path'])
        else:
            action_taken = f"F1 tidak meningkat. Kesabaran: {epoch - self.start_epoch + 1 - len([f for f in self.histories['val_f1'] if f > self.best_val_f1 + self.training_config['min_delta']])}/{self.training_config['early_stopping_patience']}."
        self.logger.info(action_taken)
        progress_data = {
            'metadata': {'last_saved_epoch': epoch, 'best_f1_score': self.best_val_f1},
            'epochs': {e: {'train_loss': tl, 'val_loss': vl, 'val_f1': vf1, 'val_acc': va} for e, (tl, vl, vf1, va) in enumerate(zip(*self.histories.values()), 1)},
            'histories': self.histories
        }
        with open(self.training_config['progress_save_path'], 'w') as f: json.dump(progress_data, f, indent=4)
        self.logger.info(f"Progres Epoch {epoch} disimpan ke JSON.")
# =============================================================================
# BAGIAN 5: FUNGSI UTAMA (MAIN) YANG RAMPING
# =============================================================================
def main():
    logger = setup_logging()
    set_seed(42)

    logger.info("--- FASE 1: MEMBUAT DAN MERESERVASI RESEP VALIDASI & TES ---")

    recipes_dir = '/content/recipes/'
    os.makedirs(recipes_dir, exist_ok=True)
    val_recipes_path = os.path.join(recipes_dir, 'validation_recipes.json')
    test_recipes_path = os.path.join(recipes_dir, 'test_recipes.json')

    if not os.path.exists(val_recipes_path):
        logger.info("Membuat resep validasi baru...")
        validation_recipes = generate_diagnostic_recipes()
        with open(val_recipes_path, 'w') as f: json.dump(validation_recipes, f, indent=2)
    else:
        logger.info(f"Memuat resep validasi dari: {val_recipes_path}")
        with open(val_recipes_path, 'r') as f: validation_recipes = json.load(f)

    if not os.path.exists(test_recipes_path):
        logger.info("Membuat resep tes baru...")
        test_recipes = generate_diagnostic_recipes()
        with open(test_recipes_path, 'w') as f: json.dump(test_recipes, f, indent=2)
    else:
        logger.info(f"Memuat resep tes dari: {test_recipes_path}")
        with open(test_recipes_path, 'r') as f: test_recipes = json.load(f)

    reserved_signatures = set()
    for r in validation_recipes + test_recipes:
        signature = (tuple(sorted(r['base_elements'])), r['temperature'], r['electron_density'])
        reserved_signatures.add(signature)

    logger.info(f"Total {len(reserved_signatures)} resep telah direservasi untuk validasi dan tes.")

    simulation_factory = SimulationFactory(SIMULATION_CONFIG, BASE_ELEMENTS, CONFIG['data']['element_map_path'])
    training_pipeline = TrainingPipeline(CONFIG, simulation_factory, validation_recipes, reserved_signatures)

    trained_model, element_map = training_pipeline.run()

    if trained_model:
        logger.info("\n" + "="*80 + "\nMEMULAI TAHAP ANALISIS FINAL\n" + "="*80)

        logger.info("Membuat dataset tes statis dari resep yang direservasi...")
        test_dataset = simulation_factory.create_dataset_from_recipes(test_recipes)

        run_final_evaluation(trained_model, element_map, CONFIG, test_dataset, "RESERVED TEST SET")

        if validation_recipes:
            sample_recipe_to_analyze = validation_recipes[10]
            analyze_from_recipe(
                model=trained_model,
                element_map=element_map,
                config=CONFIG,
                sim_factory=simulation_factory,
                recipe=sample_recipe_to_analyze,
                peak_height=0.0001,
                threshold=0.5
            )
        else:
            logger.warning("Tidak ada resep validasi untuk dianalisis secara visual.")
        
        analyze_from_asc(
            model=trained_model,
            element_map=element_map,
            config=CONFIG,
            asc_file_path=CONFIG["data"]["field_data_asc_path"],
            peak_height=0.00005,
            threshold=0.5
        )


# =============================================================================
# BAGIAN 6: BLOK EKSEKUSI UTAMA
# =============================================================================
if __name__ == '__main__':
    try:
        drive.mount('/content/drive', force_remount=True)
        print("Menyalin file data penting...")
        !cp "/content/drive/MyDrive/libs_lstm/data/processed/diagnostic_validation_set.h5" "/content/"
        !cp "/content/drive/MyDrive/libs_lstm/data/processed/element-map-18a.json" "/content/"
        !cp "/content/drive/MyDrive/libs_lstm/data/processed/nist_data(1).h5" "/content/"
        !cp "/content/drive/MyDrive/libs_lstm/data/processed/atomic_data1.h5" "/content/"
        print("Selesai menyalin.")

        main()
    except Exception as e:
        print(f"\n\nPROSES KESELURUHAN GAGAL: {e}")
        traceback.print_exc()
