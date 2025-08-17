# >>> Tambahkan kode diagnostik ini di awal sim2.py <<<
import sys
import os

print("--- DIAGNOSTIK AWAL SKRIP ---")
print(f"Versi Python yang digunakan skrip: {sys.version}")
print(f"Executable Python yang digunakan: {sys.executable}")
print(f"PYTHONPATH saat ini: {os.environ.get('PYTHONPATH', 'Tidak diatur')}")
print(f"Path pencarian modul sys.path:")
for p_idx, p_val in enumerate(sys.path): # Menggunakan enumerate untuk indeks
    print(f"  [{p_idx}] - {p_val}")      # Menampilkan indeks untuk kemudahan pembacaan

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
import os
import numpy as np
import pandas as pd
import json
import re
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
    stdout_handler.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(processName)s] %(message)s')
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    return logger

# Konfigurasi simulasi
SIMULATION_CONFIG = {
    "resolution": 4096,  # Untuk pengujian
    "wl_range": (200, 900),
    "sigma": 0.1,
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,
    "num_samples": 100000,  # MODIFIKASI: Ubah menjadi 50.000 sampel
    "temperature_range": np.linspace(5000, 25000, 100).tolist(),  # MODIFIKASI: 100 suhu dari 5000–25000 K
    "electron_density_range": np.logspace(14, 18, 100).tolist(),  # MODIFIKASI: 100 densitas dari 10^14–10^18 cm^-3
    "data_dir": "/home/bwalidain/birrulwldain/data",
    "processed_dir": "/home/bwalidain/_scratch/output",
    "logs_dir": "/home/bwalidain/birrulwldain/logs"
}

# Konstanta fisika
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Elemen dan ion
BASE_ELEMENTS = [ "Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar",  "Ti", "Mn", "Co"]  # MODIFIKASI: Tambah elemen baru, total 20
REQUIRED_ELEMENTS = [f"{elem}_{ion}" for elem in BASE_ELEMENTS for ion in [1, 2]]

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
        config: Dict = SIMULATION_CONFIG
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
        self.logger = logging.getLogger('SpectralSimulation')
        if ipex:
            torch.set_num_threads(32)  # Sesuaikan dengan cpus-per-task
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
            x_tensor = self.wavelengths.clone().detach()
            center_tensor = torch.tensor(center, device=self.device, dtype=torch.float32)
            sigma_tensor = torch.tensor(self.sigma, device=self.device, dtype=torch.float32)
            gaussian = torch.exp(-0.5 * ((x_tensor - center_tensor) / sigma_tensor) ** 2) / (sigma_tensor * torch.sqrt(torch.tensor(2 * np.pi)))
            self.gaussian_cache[center] = gaussian.cpu().numpy().astype(np.float32)
        return self.gaussian_cache[center]

    def simulate(self, atom_percentage: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]], List[List[int]], List[float], List[List[Dict]], List[np.ndarray]]:
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
                            class_idx = REQUIRED_ELEMENTS.index(self.element_label) if self.element_label in REQUIRED_ELEMENTS else len(REQUIRED_ELEMENTS)
                            peak_idx.append(idx)
                            peak_label.append(class_idx)
                except (ValueError, TypeError):
                    continue

            spectra.append(intensities.cpu().numpy())
            peak_indices.append(peak_idx)
            peak_labels.append(peak_label)
            temperatures.append(temp)
            intensity_data.append(temp_intensity_data)
            contributions.append(element_contributions.cpu().numpy())

        return self.wavelengths, spectra, peak_indices, peak_labels, temperatures, intensity_data, contributions

class MixedSpectrumSimulator:
    """Menggabungkan spektrum dari beberapa elemen berdasarkan proporsi atom."""
    def __init__(self, simulators: List[SpectrumSimulator], config: Dict, delta_E_max: Dict[str, float]):
        self.simulators = simulators
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.convolution_sigma = config["convolution_sigma"]
        self.electron_density_range = config["electron_density_range"]
        self.delta_E_max = delta_E_max
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.device = torch.device("cpu")
        self.intensity_threshold = 0.01
        self.current_T: float = 0.0
        self.current_n_e: float = 0.0
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

    def generate_sample(self, ionization_energies: Dict[str, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        self.logger.debug(f"Memulai generate_sample untuk T={self.current_T} K, n_e={self.current_n_e:.2e}")
        if self.current_n_e > 5e16 and self.current_T < 8000:
            self.logger.warning(f"n_e tinggi ({self.current_n_e:.2e} cm^-3) dan T rendah ({self.current_T} K) dapat menyebabkan self-absorption")

        temp = self.current_T
        electron_density = self.current_n_e
        num_target_elements = np.random.randint(5, 10)  # MODIFIKASI: Vary jumlah elemen 5–9
        self.logger.debug(f"Memilih {num_target_elements} elemen untuk sampel")
        selected_base_elements = np.random.choice(BASE_ELEMENTS, num_target_elements, replace=False)
        selected_pairs = [(elem, f"{elem}_1", f"{elem}_2") for elem in selected_base_elements]
        self.logger.debug("Selesai memilih elemen target")

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
            if ion_energy == 0.0:
                self.logger.warning(f"Tidak ada energi ionisasi untuk {base_elem} I")
                continue
            saha_ratio = self._saha_ratio(ion_energy, temp, electron_density)
            total_percentage = np.random.lognormal(mean=0, sigma=1.5)  # MODIFIKASI: Distribusi log-normal lebih fleksibel
            fraction_neutral = 1 / (1 + saha_ratio)
            fraction_ion = saha_ratio / (1 + saha_ratio)
            percentage_neutral = total_percentage * fraction_neutral
            percentage_ion = total_percentage * fraction_ion
            atom_percentages_dict[elem_neutral] = percentage_neutral / 100.0
            atom_percentages_dict[elem_ion] = percentage_ion / 100.0
            total_target_percentage += total_percentage
        self.logger.debug("Selesai menghitung proporsi atom")

        if total_target_percentage != 0:
            scaling_factor = 100.0 / total_target_percentage
            for key in atom_percentages_dict:
                atom_percentages_dict[key] *= scaling_factor
        else:
            self.logger.debug("Total proporsi atom nol, mengembalikan None")
            return None, None, None, None

        atom_percentages_dict['temperature'] = float(temp)
        atom_percentages_dict['electron_density'] = float(electron_density)
        atom_percentages_dict['delta_E_max'] = float(delta_E_max)

        selected_target_elements = [k for k in atom_percentages_dict.keys() if k not in ['temperature', 'electron_density', 'delta_E_max']]
        atom_percentages = np.array([atom_percentages_dict[elem] for elem in selected_target_elements], dtype=np.float32)
        selected_simulators = [sim for sim in self.simulators if f"{sim.element}_{sim.ion}" in selected_target_elements]

        if not selected_simulators:
            self.logger.warning(f"Tidak ada simulator valid untuk {selected_target_elements}")
            return None, None, None, None
        self.logger.debug("Selesai memilih simulator")

        mixed_spectrum = np.zeros(self.resolution, dtype=np.float32)
        element_contributions = np.zeros((len(selected_simulators), self.resolution), dtype=np.float32)

        for sim_idx, simulator in enumerate(selected_simulators):
            idx = selected_target_elements.index(f"{simulator.element}_{simulator.ion}")
            atom_percentage = atom_percentages[idx]
            wavelengths, element_spectra, _, _, temps, _, contributions = simulator.simulate(atom_percentage)
            for spectrum, t, contrib in zip(element_spectra, temps, contributions):
                if t == temp:
                    mixed_spectrum += spectrum
                    element_contributions[sim_idx] = contrib
                    break
        self.logger.debug("Selesai menghasilkan mixed spectrum")

        if np.max(mixed_spectrum) == 0:
            self.logger.warning(f"Tidak ada spektrum dihasilkan untuk T={temp} K")
            return None, None, None, None

        convolved_spectrum = self._convolve_spectrum(mixed_spectrum, self.convolution_sigma)
        normalized_spectrum = self._normalize_intensity(convolved_spectrum, SIMULATION_CONFIG["target_max_intensity"])
        self.logger.debug("Selesai konvolusi dan normalisasi")

        labels = np.zeros(self.resolution, dtype=np.int32)
        contributions_array = np.zeros((self.resolution, len(REQUIRED_ELEMENTS)), dtype=np.float32)

        for idx in range(self.resolution):
            contributions_at_point = element_contributions[:, idx]
            total_intensity = np.sum(contributions_at_point)
            if total_intensity >= self.intensity_threshold:
                dominant_sim_idx = np.argmax(contributions_at_point)
                dominant_element = selected_simulators[dominant_sim_idx].element_label
                if dominant_element in REQUIRED_ELEMENTS:
                    dominant_label = REQUIRED_ELEMENTS.index(dominant_element)
                    labels[idx] = dominant_label + 1
                    contributions_array[idx, dominant_label] = normalized_spectrum[idx]

        atom_percentages_dict = {k: float(v * 100) if k not in ['temperature', 'electron_density', 'delta_E_max'] else float(v)
                                for k, v in atom_percentages_dict.items()}
        self.logger.debug("Sampel selesai dihasilkan")
        return self.wavelengths, normalized_spectrum, labels, atom_percentages_dict

class DatasetGenerator:
    def __init__(self, config: Dict = SIMULATION_CONFIG):
        self.config = config
        self.temperature_range = config["temperature_range"]
        self.electron_density_range = config["electron_density_range"]
        self.num_samples = config["num_samples"]
        self.combinations_json_path = None
        self.used_combinations = set()  # Dikelola di proses utama
        self.logger = logging.getLogger('SpectralSimulation')

    def _calculate_lte_electron_density(self, temp: float, delta_E: float) -> float:
        return 1.638e12 * (temp ** 0.5) * (delta_E ** 3)

    def _hash_combination(self, temp: float, n_e: float, atom_percentages: Dict) -> str:
        elements_sorted = sorted(
            [(k, round(v, 6)) for k, v in atom_percentages.items()  # MODIFIKASI: Presisi 6 desimal
             if k not in ['temperature', 'electron_density', 'delta_E_max']],
            key=lambda x: x[0]
        )
        combination_str = f"{temp:.5f}_{n_e:.5f}_{str(elements_sorted)}"  # MODIFIKASI: Format lebih presisi
        return hashlib.sha256(combination_str.encode()).hexdigest()  # MODIFIKASI: Gunakan SHA256 untuk hash lebih kuat

    def _load_used_combinations(self) -> None:
        self.used_combinations.clear()
        if os.path.exists(self.combinations_json_path):
            try:
                with open(self.combinations_json_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        self.used_combinations.add(entry['hash'])
                self.logger.info(f"Memuat {len(self.used_combinations)} kombinasi dari {self.combinations_json_path}")
            except Exception as e:
                self.logger.error(f"Error memuat kombinasi JSON: {str(e)}")

    def _save_combinations(self, combinations: List[Dict]) -> None:
        if not self.combinations_json_path:
            return
        try:
            data = []
            if os.path.exists(self.combinations_json_path):
                with open(self.combinations_json_path, 'r') as f:
                    data = json.load(f)
            data.extend(combinations)
            with open(self.combinations_json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error menyimpan kombinasi JSON: {str(e)}")

    def _generate_sample_params(self, delta_E_max_dict: Dict[str, float]) -> List[Tuple[float, float]]:
        num_temperatures = len(self.temperature_range)
        samples_per_temp = self.num_samples // num_temperatures
        remainder = self.num_samples % num_temperatures

        temp_specific_params = {T: [] for T in self.temperature_range}
        for i, T in enumerate(self.temperature_range):
            n_samples_for_temp = samples_per_temp + (1 if i < remainder else 0)
            delta_E_values = [v for v in delta_E_max_dict.values() if v > 0]
            delta_E_max_value = max(delta_E_values) if delta_E_values else 4.0
            n_e_min = self._calculate_lte_electron_density(T, delta_E_max_value)
            valid_n_e = [n_e for n_e in self.electron_density_range if n_e >= n_e_min]
            if not valid_n_e:
                self.logger.warning(f"Tidak ada densitas elektron valid untuk T={T} K. Menggunakan seluruh rentang")
                valid_n_e = self.electron_density_range

            max_ne_per_temp = min(len(valid_n_e), 10)  # MODIFIKASI: Batasi hingga 10 densitas per suhu untuk efisiensi
            if max_ne_per_temp == 0:
                max_ne_per_temp = len(self.electron_density_range)
            selected_n_e = np.random.choice(valid_n_e, size=max_ne_per_temp, replace=False).tolist()

            if len(selected_n_e) < n_samples_for_temp:
                additional_n_e = np.random.choice(selected_n_e, size=n_samples_for_temp - len(selected_n_e), replace=True)
                selected_n_e.extend(additional_n_e)

            temp_specific_params[T].extend([(T, n_e) for n_e in selected_n_e[:n_samples_for_temp]])

        sample_params = []
        indices = {T: 0 for T in self.temperature_range}
        remaining = {T: len(params) for T, params in temp_specific_params.items()}
        total_remaining = sum(remaining.values())

        while total_remaining > 0:
            for T in self.temperature_range:
                if remaining[T] > 0:
                    idx = indices[T]
                    sample_params.append(temp_specific_params[T][idx])
                    indices[T] += 1
                    remaining[T] -= 1
                    total_remaining -= 1

        if len(sample_params) < self.num_samples:
            self.logger.warning(f"Hanya {len(sample_params)} sampel dihasilkan, menggandakan...")
            while len(sample_params) < self.num_samples:
                idx = np.random.randint(0, len(sample_params))
                sample_params.append(sample_params[idx])
        elif len(sample_params) > self.num_samples:
            sample_params = sample_params[:self.num_samples]

        return sample_params

    def _generate_sample_task(self, args: Tuple[float, float, List[SpectrumSimulator], Dict[str, float], Dict[str, float], int, set]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
        T, n_e, simulators, ionization_energies, delta_E_max, sample_id, used_combinations = args
        logger = logging.getLogger('SpectralSimulation')
        logger.debug(f"Memulai pemrosesan sampel {sample_id} dengan T={T} K, n_e={n_e:.1e} cm^-3")
        mixed_simulator = MixedSpectrumSimulator(simulators, self.config, delta_E_max)
        mixed_simulator.current_T = T
        mixed_simulator.current_n_e = n_e
        max_attempts = 50  # MODIFIKASI: Tingkatkan max_attempts untuk mendukung lebih banyak kombinasi
        attempt = 0
        sample_generated = False
        result = (None, None, None, None, None)

        while attempt < max_attempts and not sample_generated:
            result = mixed_simulator.generate_sample(ionization_energies)
            if result[0] is None:
                attempt += 1
                logger.debug(f"Percobaan {attempt}/{max_attempts} untuk sampel {sample_id} gagal")
                continue

            wavelengths, spectrum, labels, atom_percentages = result
            combination_hash = self._hash_combination(T, n_e, atom_percentages)
            if combination_hash in used_combinations:
                logger.debug(f"Kombinasi sudah ada untuk T={T} K, n_e={n_e:.1e} cm^-3, coba ulang ({attempt + 1}/{max_attempts})")
                attempt += 1
                continue
            combination = {
                'sample_id': f"sample_{sample_id}",
                'hash': combination_hash,
                'temperature': float(T),
                'electron_density': float(n_e),
                'elements': {k: v for k, v in atom_percentages.items()
                             if k not in ['temperature', 'electron_density', 'delta_E_max']},
                'delta_E_max': atom_percentages['delta_E_max']
            }
            sample_generated = True
            result = (wavelengths, spectrum, labels, atom_percentages, combination)

        if not sample_generated:
            logger.warning(f"Gagal menghasilkan sampel unik untuk T={T} K, n_e={n_e:.1e} cm^-3 setelah {max_attempts} percobaan")
            T = np.random.choice(self.temperature_range)  # MODIFIKASI: Fallback ke suhu/densitas acak
            n_e = np.random.choice(self.electron_density_range)
            logger.info(f"Falling back ke T={T} K, n_e={n_e:.1e} cm^-3 untuk sample_id={sample_id}")
            return self._generate_sample_task((T, n_e, simulators, ionization_energies, delta_E_max, sample_id, used_combinations))
        else:
            logger.debug(f"Sampel {sample_id} selesai diproses")
        return result

    def generate_dataset(
        self,
        simulators: List[SpectrumSimulator],
        delta_E_max: Dict[str, float],
        ionization_energies: Dict[str, float],
        processed_dir: str
    ) -> None:
        self.logger.info("Memulai pembuatan dataset")
        np.random.seed(42)
        self.combinations_json_path = os.path.join(processed_dir, "combinations.json")
        self._load_used_combinations()

        spectra_list, labels_list, wavelengths_list, atom_percentages_list = [], [], [], []
        sample_params = self._generate_sample_params(delta_E_max)

        temps = [param[0] for param in sample_params]
        temp_counts = Counter(temps)
        for temp in sorted(temp_counts.keys()):
            count = temp_counts[temp]
            self.logger.info(f"Distribusi suhu - Suhu {temp} K: {count} sampel ({count/self.num_samples*100:.2f}%)")

        max_workers = min(32, os.cpu_count() or 1)
        self.logger.info(f"Menggunakan {max_workers} proses untuk pembuatan dataset")

        combinations_to_save = []
        with Pool(processes=max_workers) as pool:
            tasks = [(T, n_e, simulators, ionization_energies, delta_E_max, i + 1, self.used_combinations)
                     for i, (T, n_e) in enumerate(sample_params)]
            for result in tqdm(pool.imap_unordered(self._generate_sample_task, tasks), total=len(tasks), desc="Menghasilkan dataset"):
                wavelengths, spectrum, labels, atom_percentages, combination = result
                if spectrum is not None:
                    spectra_list.append(spectrum)
                    labels_list.append(labels)
                    wavelengths_list.append(wavelengths)
                    atom_percentages_list.append(atom_percentages)
                    if combination:
                        self.used_combinations.add(combination['hash'])
                        combinations_to_save.append(combination)
                if len(spectra_list) % (self.num_samples // 5) == 0:
                    current_temps = [param[0] for param in sample_params[:len(spectra_list)]]
                    current_counts = Counter(current_temps)
                    self.logger.info(f"Progres {len(spectra_list)/self.num_samples*100:.1f}%:")
                    for temp, count in sorted(current_counts.items()):
                        self.logger.info(f"Suhu {temp} K: {count} sampel ({count/len(spectra_list)*100:.2f}%)")
                    self._save_combinations(combinations_to_save)  # MODIFIKASI: Simpan kombinasi secara batch
                    combinations_to_save = []

        self._save_combinations(combinations_to_save)

        if not spectra_list:
            self.logger.error("Tidak ada sampel valid yang dihasilkan. Periksa data NIST atau konfigurasi simulator")
            raise ValueError("Tidak ada sampel valid yang dihasilkan")

        element_counts = Counter()  # MODIFIKASI: Tambah logging distribusi elemen
        for ap in atom_percentages_list:
            elements = [k for k in ap.keys() if k not in ['temperature', 'electron_density', 'delta_E_max']]
            element_counts.update(elements)
        self.logger.info("Distribusi elemen di dataset:")
        for elem, count in element_counts.most_common():
            self.logger.info(f"{elem}: {count} sampel ({count/len(atom_percentages_list)*100:.2f}%)")

        spectra_array = np.array(spectra_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.int32)
        wavelengths_array = np.array(wavelengths_list[0], dtype=np.float32)
        atom_percentages_array = [json.dumps(d).encode('utf-8') for d in atom_percentages_list]

        num_train = int(0.7 * len(spectra_list))
        num_val = int(0.15 * len(spectra_list))
        num_test = len(spectra_list) - num_train - num_val

        indices = np.random.permutation(len(spectra_list))
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:]

        train_data = (
            spectra_array[train_idx],
            labels_array[train_idx],
            [atom_percentages_array[i] for i in train_idx]
        )
        val_data = (
            spectra_array[val_idx],
            labels_array[val_idx],
            [atom_percentages_array[i] for i in val_idx]
        )
        test_data = (
            spectra_array[test_idx],
            labels_array[test_idx],
            [atom_percentages_array[i] for i in test_idx]
        )

        output_filename = "spectral_dataset.h5"
        output_path = os.path.join(processed_dir, output_filename)
        os.makedirs(processed_dir, exist_ok=True)

        if os.path.exists(output_path):
            backup_path = os.path.join(processed_dir, f"spectral_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
            shutil.copy(output_path, backup_path)
            self.logger.info(f"Backup dibuat di {backup_path}")

        existing_data = {'train': {'spectra': [], 'labels': [], 'atom_percentages': []},
                        'validation': {'spectra': [], 'labels': [], 'atom_percentages': []},
                        'test': {'spectra': [], 'labels': [], 'atom_percentages': []}}
        total_existing_samples = 0

        if os.path.exists(output_path):
            with h5py.File(output_path, 'r') as f:
                if 'wavelengths' in f:
                    existing_wavelengths = f['wavelengths'][:]
                    if not np.array_equal(existing_wavelengths, wavelengths_array):
                        self.logger.error("Wavelengths baru tidak cocok dengan wavelengths yang sudah ada")
                        raise ValueError("Wavelengths tidak cocok")

                for ds_name in ['train', 'validation', 'test']:
                    if ds_name in f:
                        if 'spectra' in f[ds_name]:
                            existing_data[ds_name]['spectra'] = f[ds_name]['spectra'][:]
                        if 'labels' in f[ds_name]:
                            existing_data[ds_name]['labels'] = f[ds_name]['labels'][:]
                        if 'atom_percentages' in f[ds_name]:
                            existing_data[ds_name]['atom_percentages'] = f[ds_name]['atom_percentages'][:]
                        total_existing_samples += len(existing_data[ds_name]['spectra'])

        combined_data = {}
        for ds_name, new_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
            new_spectra, new_labels, new_atom_percentages = new_data
            existing_spectra = existing_data[ds_name]['spectra']
            existing_labels = existing_data[ds_name]['labels']
            existing_atom_percentages = existing_data[ds_name]['atom_percentages']

            combined_spectra = np.concatenate([existing_spectra, new_spectra]) if len(existing_spectra) > 0 else new_spectra
            combined_labels = np.concatenate([existing_labels, new_labels]) if len(existing_labels) > 0 else new_labels
            existing_atom_percentages_list = existing_atom_percentages.tolist() if isinstance(existing_atom_percentages, np.ndarray) else existing_atom_percentages
            combined_atom_percentages = existing_atom_percentages_list + new_atom_percentages if len(existing_atom_percentages_list) > 0 else new_atom_percentages

            combined_data[ds_name] = (combined_spectra, combined_labels, combined_atom_percentages)

        with h5py.File(output_path, 'a') as f:
            for ds_name in ['train', 'validation', 'test']:
                if ds_name in f:
                    del f[ds_name]

            if 'wavelengths' in f:
                del f['wavelengths']
            f.create_dataset('wavelengths', data=wavelengths_array)

            total_samples = 0
            for ds_name, (spectra, labels, atom_percentages) in combined_data.items():
                ds_grp = f.create_group(ds_name)
                ds_grp.create_dataset('spectra', data=spectra, compression='gzip')
                ds_grp.create_dataset('labels', data=labels, compression='gzip')
                ds_grp.create_dataset('atom_percentages', data=atom_percentages, compression='gzip')
                total_samples += len(spectra)

            f.attrs['last_updated'] = datetime.now().isoformat()  # MODIFIKASI: Perbaiki atribut 'lastbud' menjadi 'last_updated'
            f.attrs['total_samples'] = total_samples
            f.attrs['simulation_config'] = json.dumps(self.config)

        self.logger.info(f"Dataset di-append ke {output_path} di sub-grup train/validation/test")
        self.logger.info(f"Total sampel: {total_samples} (baru: {len(spectra_list)}, lama: {total_existing_samples})")

class DataManager:
    """Mengelola pemuatan data, validasi, dan operasi file."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "output")
        self.nist_target_path = os.path.join(self.data_dir, "nist_data(1).h5")
        self.atomic_data_target_path = os.path.join(self.data_dir, "atomic_data1.h5")
        self.json_map_path = os.path.join(self.data_dir, "element_map.json")
        self.logger = logging.getLogger('SpectralSimulation')

    def load_element_map(self) -> Dict:
        if not os.path.exists(self.json_map_path):
            self.logger.error(f"element_map.json tidak ditemukan di {self.json_map_path}")
            raise FileNotFoundError(f"element_map.json tidak ditemukan")
        with open(self.json_map_path, 'r') as f:
            element_map = json.load(f)

        for elem in REQUIRED_ELEMENTS:
            if elem not in element_map or not isinstance(element_map[elem], list) or abs(sum(element_map[elem]) - 1.0) > 1e-6:
                self.logger.error(f"Peta elemen tidak valid untuk {elem}: harus berupa daftar dengan jumlah 1.0")
                raise ValueError(f"Peta elemen tidak valid untuk {elem}")
        self.logger.info(f"Element map dimuat dengan {len(element_map)} elemen")
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

def main():
    """Fungsi utama untuk menjalankan simulasi dan menghasilkan dataset."""
    logger = setup_logging(base_dir=SIMULATION_CONFIG["logs_dir"], job_id=os.getenv("SLURM_JOB_ID", "unknown"))
    logger.info("Memulai simulasi spektral")
    if ipex:
        logger.info(f"Intel Extension for PyTorch tersedia, versi: {ipex_version}")
    else:
        logger.warning("Intel Extension for PyTorch tidak tersedia. Lanjutkan tanpa optimasi IPEX")

    pd.set_option('future.no_silent_downcasting', True)
    torch.set_num_threads(32)

    base_dir = SIMULATION_CONFIG["data_dir"]
    data_manager = DataManager(base_dir)

    element_map = data_manager.load_element_map()
    ionization_energies = data_manager.load_ionization_energies()

    fetcher = DataFetcher(data_manager.nist_target_path)
    nist_data_dict = {}
    delta_E_max_dict = {}
    for elem in REQUIRED_ELEMENTS:
        element, ion = elem.split('_')
        data, delta_E = fetcher.get_nist_data(element, int(ion))
        nist_data_dict[elem] = data
        delta_E_max_dict[elem] = delta_E
        if not data:
            logger.warning(f"Tidak ada data NIST untuk {elem}")

    simulators = []
    for elem in nist_data_dict:
        if nist_data_dict[elem]:
            element, ion = elem.split('_')
            ion_energy = ionization_energies.get(f"{element} {'I' if int(ion) == 1 else 'II'}", 0.0)
            simulator = SpectrumSimulator(
                nist_data_dict[elem],
                element,
                int(ion),
                SIMULATION_CONFIG["temperature_range"],
                ion_energy,
                SIMULATION_CONFIG
            )
            simulators.append(simulator)

    if not simulators:
        logger.error("Tidak ada simulator valid yang dibuat. Periksa data NIST")
        raise ValueError("Tidak ada simulator valid yang dibuat")

    generator = DatasetGenerator(SIMULATION_CONFIG)
    generator.generate_dataset(
        simulators,
        delta_E_max_dict,
        ionization_energies,
        SIMULATION_CONFIG["processed_dir"]
    )

    logger.info("Simulasi selesai")

if __name__ == "__main__":
    main()