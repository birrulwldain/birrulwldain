# >>> Kode diagnostik di awal sim2.py <<<
import sys
import os
import json
import re
import argparse # Pastikan import ini ada di bagian atas file Anda

print("--- DIAGNOSTIK AWAL SKRIP ---")
print(f"Versi Python yang digunakan skrip: {sys.version}")
print(f"Executable Python yang digunakan: {sys.executable}")
print(f"PYTHONPATH saat ini: {os.environ.get('PYTHONPATH', 'Tidak diatur')}")
print(f"Path pencarian modul sys.path:")
for p_idx, p_val in enumerate(sys.path):
    print(f"  [{p_idx}] - {p_val}")

try:
    print("Mencoba mengimpor IPEX di dalam skrip (awal)...")
    import intel_extension_for_pytorch as ipex_diag
    print(f"IPEX berhasil diimpor di skrip (awal)! Versi: {ipex_diag.__version__}")
except ImportError as e:
    print(f"IPEX GAGAL diimpor di dalam skrip (awal). Error: {e}")
except Exception as e:
    print(f"Error lain saat mencoba impor IPEX di skrip (awal): {e}")
print("--- AKHIR DIAGNOSTIK AWAL SKRIP ---")
print("\nMelanjutkan eksekusi skrip asli...\n")
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import h5py
from tqdm import tqdm
import shutil
from typing import List, Dict, Tuple, Optional
from collections import Counter
import hashlib
from datetime import datetime
import logging
from multiprocessing import Pool

# Impor IPEX dengan logging versi
try:
    import intel_extension_for_pytorch as ipex
    ipex_version = getattr(ipex, '__version__', 'unknown')
except ImportError:
    ipex = None
    ipex_version = 'not installed'

# Konfigurasi logging
def setup_logging(base_dir: str, job_id: str = "unknown"):
    """Konfigurasi logger untuk stdout dan file dengan format terstruktur."""
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"spectral_simulation_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('SpectralSimulation')
    logger.setLevel(logging.DEBUG)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(processName)s] %(message)s')
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    return logger

# Konfigurasi simulasi
SIMULATION_CONFIG = {
    "resolution": 4096,
    "wl_range": (200, 900),
    "sigma": 0.1,
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,
    "num_samples": 2500,
    "temperature_range": np.linspace(5000, 25000, 100).tolist(),
    "electron_density_range": np.logspace(14, 18, 100).tolist(),
    "data_dir": "/home/bwalidain/birrulwldain/data",
    "processed_dir": "/home/bwalidain/_scratch/",
    "logs_dir": "/home/bwalidain/birrulwldain/logs"
}

# Konstanta fisika
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Elemen dan ion
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]
class DataFetcher:
    """Mengambil data spektral dari file HDF5 NIST untuk elemen dan ion tertentu."""
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path
        self.delta_E_max: Dict[str, float] = {}
        self.missing_data_count: Dict[str, int] = {}
        self.logger = logging.getLogger('SpectralSimulation')

    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        elem_key = f"{element}_{sp_num}"
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                filtered_df = df[(df['element'] == element) & (df['sp_num'] == sp_num)]
                required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']

                if filtered_df.empty or not all(col in df.columns for col in required_columns):
                    self.missing_data_count[elem_key] = self.missing_data_count.get(elem_key, 0) + 1
                    if self.missing_data_count[elem_key] <= 3:
                        self.logger.warning(f"Tidak ada data untuk {elem_key} di dataset NIST")
                    return [], 0.0

                filtered_df = filtered_df.dropna(subset=required_columns)

                filtered_df['ritz_wl_air(nm)'] = pd.to_numeric(filtered_df['ritz_wl_air(nm)'], errors='coerce')
                for col in ['Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k']:
                    filtered_df[col] = pd.to_numeric(
                        filtered_df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)) if re.sub(r'[^\d.-]', '', str(x)) else None),
                        errors='coerce'
                    )

                filtered_df = filtered_df.dropna(subset=['ritz_wl_air(nm)', 'Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k'])

                filtered_df = filtered_df[
                    (filtered_df['ritz_wl_air(nm)'] >= SIMULATION_CONFIG["wl_range"][0]) &
                    (filtered_df['ritz_wl_air(nm)'] <= SIMULATION_CONFIG["wl_range"][1])
                ]

                filtered_df['delta_E'] = abs(filtered_df['Ek(eV)'] - filtered_df['Ei(eV)'])
                if filtered_df.empty:
                    self.missing_data_count[elem_key] = self.missing_data_count.get(elem_key, 0) + 1
                    if self.missing_data_count[elem_key] <= 3:
                        self.logger.warning(f"Tidak ada transisi valid untuk {elem_key} di rentang panjang gelombang")
                    return [], 0.0

                filtered_df = filtered_df.sort_values(by='Aki(s^-1)', ascending=False)
                delta_E_max = filtered_df['delta_E'].max()
                delta_E_max = 0.0 if pd.isna(delta_E_max) else delta_E_max
                self.delta_E_max[elem_key] = delta_E_max

                return filtered_df[required_columns + ['Acc']].values.tolist(), delta_E_max
        except Exception as e:
            self.logger.error(f"Error mengambil data NIST untuk {elem_key}: {str(e)}")
            return [], 0.0

class SpectrumSimulator:
    """Mensimulasikan spektrum emisi untuk satu elemen dan ion pada berbagai suhu."""
    def __init__(
        self,
        nist_data: List[List],
        element: str,
        ion: int,
        temperatures: List[float],
        ionization_energy: float,
        config: Dict = SIMULATION_CONFIG,
        element_map_labels: Dict[str, List[float]] = None
    ):
        self.nist_data = nist_data
        self.element = element
        self.ion = ion
        self.temperatures = temperatures
        self.ionization_energy = ionization_energy
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.sigma = config["sigma"]
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.gaussian_cache: Dict[float, np.ndarray] = {}
        self.element_label = f"{element}_{ion}"
        self.device = torch.device("cpu")
        self.element_map_labels = element_map_labels or {}  # Menyimpan mapping one-hot dari element_map.json
        self.logger = logging.getLogger('SpectralSimulation')
        if ipex:
            torch.set_num_threads(4)
            self.wavelengths = torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)
            self.logger.debug(f"IPEX dimuat, versi: {ipex_version}")

    def _partition_function(self, energy_levels: List[float], degeneracies: List[float], temperature: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return sum(g * np.exp(-E / (k_B * temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0

    def _calculate_intensity(self, temperature: float, energy: float, degeneracy: float, einstein_coeff: float, Z: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * temperature))) / Z

    def _gaussian_profile(self, center: float) -> np.ndarray:
        if center not in self.gaussian_cache:
            x_tensor = self.wavelengths.clone().detach() if isinstance(self.wavelengths, torch.Tensor) else torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)
            center_tensor = torch.tensor(center, device=self.device, dtype=torch.float32)
            sigma_tensor = torch.tensor(self.sigma, device=self.device, dtype=torch.float32)
            gaussian = torch.exp(-0.5 * ((x_tensor - center_tensor) / sigma_tensor) ** 2) / (sigma_tensor * torch.sqrt(torch.tensor(2 * np.pi)))
            self.gaussian_cache[center] = gaussian.cpu().numpy().astype(np.float32)
        return self.gaussian_cache[center]

    def simulate(self, atom_percentage: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]], List[List[List[float]]], List[float], List[List[Dict]], List[np.ndarray]]:
        if not self.nist_data:
            self.logger.warning(f"Tidak ada data NIST untuk {self.element_label}, melewati simulasi")
            return self.wavelengths, [], [], [], [], [], []

        spectra, peak_indices, peak_labels, temperatures, intensity_data, contributions = [], [], [], [], [], []
        levels = {}

        for data in self.nist_data:
            try:
                wl, Aki, Ek, Ei, gi, gk, _ = data
                if all(v is not None for v in [wl, Aki, Ek, Ei, gi, gk]):
                    Ek = float(Ek)
                    Ei = float(Ei)
                    if Ei not in levels:
                        levels[Ei] = float(gi)
                    if Ek not in levels:
                        levels[Ek] = float(gk)
            except (ValueError, TypeError):
                continue

        if not levels:
            self.logger.warning(f"Tidak ada tingkat energi valid untuk {self.element_label}")
            return self.wavelengths, [], [], [], [], [], []

        energy_levels = list(levels.keys())
        degeneracies = list(levels.values())
        num_labels = len(self.element_map_labels) if self.element_map_labels else 1
        default_label = [0.0] * num_labels  # Vektor nol untuk label default

        for temp in self.temperatures:
            Z = self._partition_function(energy_levels, degeneracies, temp)
            intensities = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
            element_contributions = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
            peak_idx, peak_label, temp_intensity_data = [], [], []

            for data in self.nist_data:
                try:
                    wl, Aki, Ek, Ei, gi, gk, _ = data
                    if all(v is not None for v in [wl, Aki, Ek, Ei, gi, gk]):
                        wl = float(wl)
                        Aki = float(Aki)
                        Ek = float(Ek)
                        intensity = self._calculate_intensity(temp, Ek, float(gk), Aki, Z)
                        idx = np.searchsorted(self.wavelengths, wl)
                        if 0 <= idx < self.resolution:
                            gaussian_contribution = torch.tensor(
                                intensity * atom_percentage * self._gaussian_profile(wl),
                                device=self.device,
                                dtype=torch.float32
                            )
                            start_idx = max(0, idx - len(gaussian_contribution) // 2)
                            end_idx = min(self.resolution, start_idx + len(gaussian_contribution))
                            if start_idx < end_idx:
                                intensities[start_idx:end_idx] += gaussian_contribution[:end_idx - start_idx]
                                element_contributions[start_idx:end_idx] += gaussian_contribution[:end_idx - start_idx]
                            temp_intensity_data.append({
                                'wavelength': wl,
                                'intensity': intensity * atom_percentage,
                                'element_label': self.element_label,
                                'index': idx
                            })
                            # Gunakan vektor one-hot dari element_map.json
                            one_hot_label = self.element_map_labels.get(self.element_label, default_label)
                            peak_idx.append(idx)
                            peak_label.append(one_hot_label)
                except (ValueError, TypeError):
                    continue

            spectra.append(intensities.cpu().numpy())
            peak_indices.append(peak_idx)
            peak_labels.append(peak_label)
            temperatures.append(temp)
            intensity_data.append(temp_intensity_data)
            contributions.append(element_contributions.cpu().numpy())

        return self.wavelengths, spectra, peak_indices, peak_labels, temperatures, intensity_data, contributions
    def simulate_single_temp(self, temp: float, atom_percentage: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """
            Mensimulasikan spektrum hanya untuk satu suhu spesifik secara efisien.
            Mengembalikan spektrum intensitas dan spektrum kontribusi elemen.
            """
            if not self.nist_data:
                # Tidak perlu logging di sini karena sudah ada di simulate() jika dipanggil
                return None
            
            # 1. Dapatkan semua tingkat energi dan degenerasi sekali saja
            levels = {}
            for data in self.nist_data:
                try:
                    # Kolom ke-2 hingga ke-5 adalah Ek, Ei, gi, gk
                    _, _, Ek, Ei, gi, gk, _ = data
                    if all(v is not None for v in [Ek, Ei, gi, gk]):
                        levels[float(Ei)] = float(gi)
                        levels[float(Ek)] = float(gk)
                except (ValueError, TypeError):
                    continue

            if not levels:
                self.logger.warning(f"Tidak ada tingkat energi valid untuk {self.element_label} pada suhu {temp}K.")
                return None

            energy_levels = list(levels.keys())
            degeneracies = list(levels.values())

            # 2. Langsung hitung untuk suhu yang diberikan (TIDAK ADA LOOP)
            Z = self._partition_function(energy_levels, degeneracies, temp)
            
            # Siapkan tensor untuk menampung hasil
            intensities = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
            element_contributions = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)

            # 3. Loop melalui setiap garis spektral untuk membangun spektrum
            for data in self.nist_data:
                try:
                    wl, Aki, Ek, Ei, gi, gk, _ = data
                    if all(v is not None for v in [wl, Aki, Ek, Ei, gi, gk]):
                        wl = float(wl)
                        Aki = float(Aki)
                        Ek = float(Ek)
                        
                        intensity = self._calculate_intensity(temp, Ek, float(gk), Aki, Z)
                        idx = np.searchsorted(self.wavelengths, wl)

                        if 0 <= idx < self.resolution:
                            # Buat profil Gaussian untuk garis ini dan tambahkan ke spektrum total
                            gaussian_contribution = torch.tensor(
                                intensity * atom_percentage * self._gaussian_profile(wl),
                                device=self.device,
                                dtype=torch.float32
                            )
                            start_idx = max(0, idx - len(gaussian_contribution) // 2)
                            end_idx = min(self.resolution, start_idx + len(gaussian_contribution))
                            
                            if start_idx < end_idx:
                                # Tambahkan kontribusi ke spektrum gabungan dan spektrum kontribusi individu
                                intensities[start_idx:end_idx] += gaussian_contribution[:end_idx - start_idx]
                                element_contributions[start_idx:end_idx] += gaussian_contribution[:end_idx - start_idx]
                except (ValueError, TypeError):
                    continue
            
            # 4. Kembalikan hasilnya sebagai numpy array
            return intensities.cpu().numpy(), element_contributions.cpu().numpy()
class MixedSpectrumSimulator:
    def __init__(self, simulators: List[SpectrumSimulator], config: Dict, delta_E_max: Dict[str, float], element_map_labels: Dict[str, List[float]]):
        self.simulators = simulators
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.convolution_sigma = config["convolution_sigma"]
        self.electron_density_range = config["electron_density_range"]
        self.delta_E_max = delta_E_max
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.device = torch.device("cpu")
        self.intensity_threshold = 0.001 # Menggunakan threshold yang lebih rendah
        self.current_T: float = 0.0
        self.current_n_e: float = 0.0
        self.element_map_labels = element_map_labels
        self.num_labels = len(next(iter(element_map_labels.values())))
        self.logger = logging.getLogger('SpectralSimulation')

    def _normalize_intensity(self, intensity: np.ndarray, target_max: float) -> np.ndarray:
        intensity_tensor = torch.tensor(intensity, device=self.device, dtype=torch.float32)
        max_intensity = torch.max(torch.abs(intensity_tensor))
        if max_intensity == 0:
            return intensity
        return (intensity_tensor / max_intensity * target_max).cpu().numpy()

    def _convolve_spectrum(self, spectrum: np.ndarray, sigma_nm: float) -> np.ndarray:
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32, device=self.device)
        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = sigma_nm / wavelength_step
        kernel_size = int(6 * sigma_points) | 1
        kernel = torch.tensor(
            gaussian(kernel_size, sigma_points) / np.sum(gaussian(kernel_size, sigma_points)),
            device=self.device,
            dtype=torch.float32
        )
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        spectrum_tensor = spectrum_tensor.unsqueeze(0).unsqueeze(0)
        convolved = F.conv1d(spectrum_tensor, kernel, padding=kernel_size // 2).squeeze().cpu().numpy()
        return convolved.astype(np.float32)

    def _saha_ratio(self, ion_energy: float, temp: float, electron_density: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        m_e = PHYSICAL_CONSTANTS["m_e"]
        h = PHYSICAL_CONSTANTS["h"]
        two_pi_me_kT_h2 = (2 * np.pi * m_e * (k_B * temp * 1.60217662e-16) / (h * 1.60217662e-19) ** 2) ** (3/2)
        two_pi_me_kT_h2 /= 1e6
        U_i = 1.0
        U_ip1 = 1.0
        saha_factor = (2 * U_ip1 / U_i) * two_pi_me_kT_h2 / electron_density
        return saha_factor * np.exp(-ion_energy / (k_B * temp))

    def generate_sample(self, ionization_energies: Dict[str, float], selected_base_elements: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        self.logger.debug(f"Menjalankan resep: T={self.current_T} K, n_e={self.current_n_e:.2e}, elemen={selected_base_elements}")

        temp = self.current_T
        electron_density = self.current_n_e
        
        selected_pairs = [(elem, f"{elem}_1", f"{elem}_2") for elem in selected_base_elements if f"{elem}_1" in REQUIRED_ELEMENTS and f"{elem}_2" in REQUIRED_ELEMENTS]
        
        if not selected_pairs:
            self.logger.warning("Tidak ada pasangan ion valid dari elemen yang dipilih.")
            return None

        delta_E_values = []
        for base_elem, elem_neutral, elem_ion in selected_pairs:
            for elem in [elem_neutral, elem_ion]:
                delta_E = self.delta_E_max.get(elem, 0.0)
                if delta_E > 0.0:
                    delta_E_values.append(delta_E)
        delta_E_max = max(delta_E_values) if delta_E_values else 4.0

        atom_percentages_dict = {}
        total_target_percentage = 0.0
        for base_elem, elem_neutral, elem_ion in selected_pairs:
            ion_energy = ionization_energies.get(f"{base_elem} I", 0.0)
            saha_ratio = self._saha_ratio(ion_energy, temp, electron_density)
            total_percentage = np.random.lognormal(mean=0, sigma=1.5)
            fraction_neutral = 1 / (1 + saha_ratio)
            fraction_ion = saha_ratio / (1 + saha_ratio)
            percentage_neutral = total_percentage * fraction_neutral
            percentage_ion = total_percentage * fraction_ion
            atom_percentages_dict[elem_neutral] = percentage_neutral / 100.0
            atom_percentages_dict[elem_ion] = percentage_ion / 100.0
            total_target_percentage += total_percentage
        
        if total_target_percentage > 0:
            scaling_factor = 100.0 / total_target_percentage
            for key in atom_percentages_dict:
                atom_percentages_dict[key] *= scaling_factor
        else:
            self.logger.warning("Total persentase atom adalah nol.")
            return None

        atom_percentages_dict['temperature'] = float(temp)
        atom_percentages_dict['electron_density'] = float(electron_density)
        atom_percentages_dict['delta_E_max'] = float(delta_E_max)

        selected_target_elements = [k for k in atom_percentages_dict.keys() if k not in ['temperature', 'electron_density', 'delta_E_max']]
        atom_percentages = np.array([atom_percentages_dict.get(e, 0) for e in selected_target_elements], dtype=np.float32)
        selected_simulators = [sim for sim in self.simulators if f"{sim.element}_{sim.ion}" in selected_target_elements]

        if not selected_simulators:
            self.logger.warning(f"Tidak ada simulator valid untuk elemen {selected_target_elements}")
            return None

        mixed_spectrum = np.zeros(self.resolution, dtype=np.float32)
        element_contributions = np.zeros((len(selected_simulators), self.resolution), dtype=np.float32)

        for sim_idx, simulator in enumerate(selected_simulators):
            idx_in_list = selected_target_elements.index(f"{simulator.element}_{simulator.ion}")
            atom_percentage = atom_percentages[idx_in_list]
            
            # >>> PERUBAHAN OPTIMASI UTAMA <<<
            # Panggil fungsi baru yang efisien, langsung berikan 'temp' dari resep
            result = simulator.simulate_single_temp(temp, atom_percentage)
            
            if result is not None:
                spectrum, contrib = result
                mixed_spectrum += spectrum
                element_contributions[sim_idx] = contrib

        if np.max(mixed_spectrum) == 0:
            self.logger.warning(f"Tidak ada spektrum dihasilkan untuk resep ini (T={temp}K).")
            return None

        convolved_spectrum = self._convolve_spectrum(mixed_spectrum, self.convolution_sigma)
        normalized_spectrum = self._normalize_intensity(convolved_spectrum, SIMULATION_CONFIG["target_max_intensity"])

        labels = np.zeros((self.resolution, self.num_labels), dtype=np.float32)
        background_label = np.array(self.element_map_labels["background"], dtype=np.float32)
        for idx in range(self.resolution):
            contributions_at_point = element_contributions[:, idx]
            total_intensity = np.sum(contributions_at_point)
            if total_intensity >= self.intensity_threshold:
                dominant_sim_idx = np.argmax(contributions_at_point)
                dominant_element = selected_simulators[dominant_sim_idx].element_label
                one_hot_label = np.array(self.element_map_labels.get(dominant_element, background_label), dtype=np.float32)
            else:
                one_hot_label = background_label
            labels[idx] = one_hot_label

        final_atom_percentages = {k: float(v * 100) if k not in ['temperature', 'electron_density', 'delta_E_max'] else float(v)
                                for k, v in atom_percentages_dict.items()}
        
        return self.wavelengths, normalized_spectrum, labels, final_atom_percentages
    
class WorkerDatasetGenerator:
    """Kelas ini mengelola eksekusi resep oleh proses-proses pekerja."""
    def __init__(self, config: Dict, element_map_labels: Dict[str, List[float]]):
        self.config = config
        self.element_map_labels = element_map_labels
        self.logger = logging.getLogger('SpectralSimulation')

    def _generate_task(self, args: Tuple) -> Optional[Tuple]:
        """Fungsi pembungkus untuk menjalankan satu resep oleh satu proses."""
        recipe, simulators, ionization_energies, delta_E_max_dict, element_map_labels = args
        
        try:
            # Setiap proses membuat instance simulator-nya sendiri
            mixed_simulator = MixedSpectrumSimulator(simulators, self.config, delta_E_max_dict, element_map_labels)
            mixed_simulator.current_T = recipe["temperature"]
            mixed_simulator.current_n_e = recipe["electron_density"]
            
            result = mixed_simulator.generate_sample(
                ionization_energies,
                recipe["base_elements"]
            )
            
            if result is None:
                self.logger.warning(f"Gagal menghasilkan spektrum untuk resep ID: {recipe['sample_id']}")
                return None
            
            wavelengths, spectrum, labels, atom_percentages = result
            # Sertakan metadata asli untuk disimpan
            atom_percentages['sample_id'] = recipe['sample_id']
            return spectrum, labels, atom_percentages
        except Exception as e:
            self.logger.error(f"Error pada task untuk resep ID {recipe.get('sample_id', 'N/A')}: {e}", exc_info=True)
            return None

    def process_recipes(self, recipes: List[Dict], simulators: List[SpectrumSimulator], ionization_energies: Dict[str, float], delta_E_max_dict: Dict[str, float], output_h5_path: str):
        """Memproses daftar resep dan menyimpan hasilnya ke file HDF5."""
        self.logger.info(f"Akan memproses {len(recipes)} resep dan menyimpan ke {output_h5_path}")

        spectra_list, labels_list, atom_percentages_list = [], [], []
        
        # Atur jumlah worker sesuai core yang tersedia atau batasan
        max_workers = min(16, os.cpu_count() or 1)
        self.logger.info(f"Menggunakan {max_workers} proses pekerja.")
        
        with Pool(processes=max_workers) as pool:
            tasks = [(recipe, simulators, ionization_energies, delta_E_max_dict, self.element_map_labels) for recipe in recipes]
            
            output_filename = os.path.basename(output_h5_path)
            for result in tqdm(pool.imap_unordered(self._generate_task, tasks), total=len(tasks), desc=f"Processing {output_filename}"):
                if result is not None:
                    spectrum, labels, atom_percentages = result
                    spectra_list.append(spectrum)
                    labels_list.append(labels)
                    atom_percentages_list.append(atom_percentages)
        
        if not spectra_list:
            self.logger.error("Tidak ada sampel valid yang dihasilkan dari resep yang diberikan.")
            return

        self.logger.info(f"Menyimpan {len(spectra_list)} spektrum yang dihasilkan...")
        with h5py.File(output_h5_path, 'w') as f:
            # Simpan wavelengths, cukup sekali
            wavelengths_data = np.linspace(self.config["wl_range"][0], self.config["wl_range"][1], self.config["resolution"], dtype=np.float32)
            f.create_dataset('wavelengths', data=wavelengths_data)
            
            # Simpan semua data ke dalam satu grup untuk digabung nanti
            data_grp = f.create_group('chunk_data')
            data_grp.create_dataset('spectra', data=np.array(spectra_list, dtype=np.float32), compression='gzip')
            data_grp.create_dataset('labels', data=np.array(labels_list, dtype=np.float32), compression='gzip')
            
            atom_percentages_encoded = [json.dumps(d).encode('utf-8') for d in atom_percentages_list]
            dt = h5py.special_dtype(vlen=str)
            data_grp.create_dataset('atom_percentages', data=np.array(atom_percentages_encoded), dtype=dt, compression='gzip')

        self.logger.info(f"Pekerjaan selesai. Data disimpan di {output_h5_path}")

class DataManager:
    """Mengelola pemuatan data, validasi, dan operasi file."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "output")
        self.nist_target_path = os.path.join(self.data_dir, "nist_data(1).h5")
        self.atomic_data_target_path = os.path.join(self.data_dir, "atomic_data1.h5")
        self.json_map_path = os.path.join(self.data_dir, "element-map-35.json")
        self.logger = logging.getLogger('SpectralSimulation')

    def load_element_map(self) -> Dict[str, List[float]]:
        if not os.path.exists(self.json_map_path):
            self.logger.error(f"element_map.json tidak ditemukan di {self.json_map_path}")
            raise FileNotFoundError(f"element_map.json tidak ditemukan")
        with open(self.json_map_path, 'r') as f:
            element_map = json.load(f)

        # Validasi bahwa setiap elemen memiliki vektor one-hot yang valid
        vector_length = None
        for elem, vector in element_map.items():
            if not isinstance(vector, list) or not all(isinstance(v, (int, float)) for v in vector):
                self.logger.error(f"Vektor one-hot tidak valid untuk {elem}: {vector}")
                raise ValueError(f"Vektor one-hot tidak valid untuk {elem}")
            if vector_length is None:
                vector_length = len(vector)
            elif len(vector) != vector_length:
                self.logger.error(f"Panjang vektor one-hot tidak konsisten untuk {elem}: {len(vector)} vs {vector_length}")
                raise ValueError(f"Panjang vektor one-hot tidak konsisten")
        self.logger.info(f"Element map dimuat dengan {len(element_map)} elemen, panjang vektor one-hot: {vector_length}")
        return element_map

    def load_ionization_energies(self) -> Dict[str, float]:
        ionization_energies = {}
        try:
            with h5py.File(self.atomic_data_target_path, 'r') as f:
                dset = f['elements']
                columns = dset.attrs.get('columns', ['At. num', 'Sp. Name', 'Ion Charge', 'El. Name',
                                                   'Prefix', 'Ionization Energy (eV)', 'Suffix'])
                data = [[item[0], item[1].decode('utf-8'), item[2].decode('utf-8'), item[3].decode('utf-8'),
                        item[4].decode('utf-8'), item[5], item[6].decode('utf-8')] for item in dset[:]]
                df_ionization = pd.DataFrame(data, columns=columns)

                species_col = ion_energy_col = None
                for col in columns:
                    if col.lower() in ['sp.', 'species', 'sp', 'element', 'sp. name']:
                        species_col = col
                    if 'ionization' in col.lower() and 'ev' in col.lower():
                        ion_energy_col = col

                if not species_col or not ion_energy_col:
                    self.logger.error(f"Kolom yang diperlukan tidak ditemukan. Tersedia: {list(df_ionization.columns)}")
                    raise KeyError(f"Kolom yang diperlukan tidak ditemukan")

                for _, row in df_ionization.iterrows():
                    try:
                        ionization_energies[row[species_col]] = float(row[ion_energy_col])
                    except (ValueError, TypeError):
                        self.logger.warning(f"Energi ionisasi tidak valid untuk {row[species_col]}, menggunakan 0.0 eV")
                        ionization_energies[row[species_col]] = 0.0

        except Exception as e:
            self.logger.error(f"Error memuat atomic_data1.h5: {str(e)}")
            raise

        for elem in REQUIRED_ELEMENTS:
            base_elem, ion = elem.split('_')
            ion_level = 'I' if ion == '1' else 'II'
            sp_name = f"{base_elem} {ion_level}"
            if sp_name not in ionization_energies:
                self.logger.warning(f"Tidak ada energi ionisasi untuk {sp_name}, menggunakan 0.0 eV")
                ionization_energies[sp_name] = 0.0

        return ionization_energies


def main(args):
    """Fungsi utama untuk menjalankan simulasi dalam mode pekerja."""
    
    # Membuat ID unik untuk file log berdasarkan nama file output
    job_id = os.path.basename(args.output_h5).replace('.h5', '')
    logger = setup_logging(base_dir=SIMULATION_CONFIG["logs_dir"], job_id=job_id)
    logger.info(f"Memulai PEKERJA simulasi spektral untuk file resep: {args.input_json}")

    # PENTING: Set jumlah thread ke 1 untuk multiprocessing
    # Paralelisme datang dari jumlah proses (worker), bukan dari thread di dalam proses.
    # Ini mencegah perebutan sumber daya CPU dan membuat kinerja lebih efisien.
    torch.set_num_threads(4)

    # 1. Muat semua data pendukung yang diperlukan
    logger.info("Memuat data pendukung (NIST, Ionization, Element Map)...")
    base_dir = SIMULATION_CONFIG["data_dir"]
    data_manager = DataManager(base_dir)
    element_map = data_manager.load_element_map()
    ionization_energies = data_manager.load_ionization_energies()
    
    fetcher = DataFetcher(data_manager.nist_target_path)
    nist_data_dict = {}
    delta_E_max_dict = {}
    for elem in tqdm(REQUIRED_ELEMENTS, desc="Fetching NIST Data"):
        element, ion = elem.split('_')
        data, delta_E = fetcher.get_nist_data(element, int(ion))
        nist_data_dict[elem] = data
        delta_E_max_dict[elem] = delta_E

    # 2. Buat instance simulator untuk setiap ion
    # Kita tidak perlu lagi memberikan rentang suhu penuh, karena setiap task akan
    # mendapatkan suhu spesifik dari resep. Beri list kosong saja.
    logger.info("Mempersiapkan simulator untuk setiap ion...")
    simulators = []
    for elem_key, nist_data in nist_data_dict.items():
        if nist_data:
            element, ion_str = elem_key.split('_')
            ion = int(ion_str)
            ion_name_suffix = 'I' if ion == 1 else 'II'
            ion_energy = ionization_energies.get(f"{element} {ion_name_suffix}", 0.0)
            simulator = SpectrumSimulator(
                nist_data=nist_data, 
                element=element, 
                ion=ion, 
                temperatures=SIMULATION_CONFIG["temperature_range"], # <-- GUNAKAN RENTANG SUHU LENGKAP
                ionization_energy=ion_energy, 
                config=SIMULATION_CONFIG, 
                element_map_labels=element_map
            )
            simulators.append(simulator)
    logger.info(f"Total {len(simulators)} simulator berhasil dibuat.")

    # 3. Baca resep dari file JSON yang menjadi jatah pekerjaan ini
    logger.info(f"Membaca resep dari {args.input_json}...")
    try:
        with open(args.input_json, 'r') as f:
            recipes = json.load(f)
    except FileNotFoundError:
        logger.error(f"File resep tidak ditemukan: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Gagal mem-parsing file JSON: {args.input_json}. Pastikan formatnya benar.")
        sys.exit(1)

    # 4. Jalankan generator dalam mode pekerja
    generator = WorkerDatasetGenerator(SIMULATION_CONFIG, element_map)
    generator.process_recipes(
        recipes=recipes, 
        simulators=simulators, 
        ionization_energies=ionization_energies, 
        delta_E_max_dict=delta_E_max_dict, 
        output_h5_path=args.output_h5
    )

    logger.info("Proses pekerja selesai.")

if __name__ == "__main__":
    # Setup untuk membaca argumen dari command line
    parser = argparse.ArgumentParser(description="Simulasi Spektral - Mode Pekerja")
    parser.add_argument('--input-json', type=str, required=True, 
                        help='Path ke file JSON berisi resep (jatah pekerjaan)')
    parser.add_argument('--output-h5', type=str, required=True, 
                        help='Path ke file HDF5 untuk menyimpan hasil')
    
    parsed_args = parser.parse_args()
    
    # Panggil fungsi main dengan argumen yang sudah di-parse
    main(parsed_args)
