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
try:
    import intel_extension_for_pytorch as ipex  # Optimasi untuk CPU Intel
except ImportError:
    ipex = None
    print("Peringatan: Intel Extension for PyTorch tidak tersedia. Lanjutkan tanpa optimasi IPEX.")

# Konfigurasi simulasi
SIMULATION_CONFIG = {
    "resolution": 2048,  # Dikurangi untuk efisiensi CPU
    "wl_range": (200, 900),
    "sigma": 0.1,  # nm, untuk pelebaran Gaussian
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,  # nm, untuk konvolusi spektrum
    "num_samples": 500,  # Dikurangi untuk uji coba
    "temperature_range": [6000, 8000, 10000, 12000, 14000, 15000],  # K
    "electron_density_range": np.logspace(15, 17, 10).tolist(),  # cm^-3
}

# Konstanta fisika
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Elemen dan ion
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl"]
REQUIRED_ELEMENTS = [f"{elem}_{ion}" for elem in BASE_ELEMENTS for ion in [1, 2]]

class DataFetcher:
    """Mengambil data spektral dari file HDF5 NIST untuk elemen dan ion tertentu."""
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path
        self.delta_E_max: Dict[str, float] = {}

    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                filtered_df = df[(df['element'] == element) & (df['sp_num'] == sp_num)]
                required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']

                if filtered_df.empty or not all(col in df.columns for col in required_columns):
                    print(f"Tidak ada data untuk {element}_{sp_num} di dataset NIST")
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
                    print(f"Tidak ada transisi valid untuk {element}_{sp_num} di rentang panjang gelombang")
                    return [], 0.0

                filtered_df = filtered_df.sort_values(by='Aki(s^-1)', ascending=False)
                delta_E_max = filtered_df['delta_E'].max()
                delta_E_max = 0.0 if pd.isna(delta_E_max) else delta_E_max
                self.delta_E_max[f"{element}_{sp_num}"] = delta_E_max

                return filtered_df[required_columns + ['Acc']].values.tolist(), delta_E_max
        except Exception as e:
            print(f"Error mengambil data NIST untuk {element}_{sp_num}: {str(e)}")
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
        self.device = torch.device("cpu")  # Paksa penggunaan CPU
        if ipex:
            # Optimasi untuk CPU Intel
            torch.set_num_threads(16)  # Sesuaikan dengan jumlah core HPC
            self.wavelengths = torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)

    def _partition_function(self, energy_levels: List[float], degeneracies: List[float], temperature: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return sum(g * np.exp(-E / (k_B * temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0

    def _calculate_intensity(self, temperature: float, energy: float, degeneracy: float, einstein_coeff: float, Z: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * temperature))) / Z

    def _gaussian_profile(self, center: float) -> np.ndarray:
        if center not in self.gaussian_cache:
            x_tensor = torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)
            center_tensor = torch.tensor(center, device=self.device, dtype=torch.float32)
            sigma_tensor = torch.tensor(self.sigma, device=self.device, dtype=torch.float32)
            with torch.cpu.amp.autocast():  # Gunakan bf16 untuk efisiensi
                gaussian = torch.exp(-0.5 * ((x_tensor - center_tensor) / sigma_tensor) ** 2) / (sigma_tensor * torch.sqrt(torch.tensor(2 * np.pi)))
            self.gaussian_cache[center] = gaussian.cpu().numpy().astype(np.float32)
        return self.gaussian_cache[center]

    def simulate(self, atom_percentage: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]], List[List[int]], List[float], List[List[Dict]], List[np.ndarray]]:
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
            print(f"Peringatan: Tidak ada tingkat energi valid untuk {self.element_label}")
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
                                with torch.cpu.amp.autocast():  # Optimasi bf16
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
        self.device = torch.device("cpu")  # Paksa penggunaan CPU
        self.intensity_threshold = 0.01
        self.current_T: float = 0.0
        self.current_n_e: float = 0.0

    def _normalize_intensity(self, intensity: np.ndarray, target_max: float) -> np.ndarray:
        intensity_tensor = torch.tensor(intensity, device=self.device, dtype=torch.float32)
        max_intensity = torch.max(torch.abs(intensity_tensor))
        if max_intensity == 0:
            return intensity
        with torch.cpu.amp.autocast():
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
        with torch.cpu.amp.autocast():
            convolved = F.conv1d(spectrum_tensor, kernel, padding=kernel_size//2).squeeze().cpu().numpy()
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
        temp = self.current_T
        electron_density = self.current_n_e
        num_target_elements = 7
        selected_base_elements = np.random.choice(BASE_ELEMENTS, num_target_elements, replace=False)
        selected_pairs = [(elem, f"{elem}_1", f"{elem}_2") for elem in selected_base_elements]

        delta_E_values = []
        for base_elem, elem_neutral, elem_ion in selected_pairs:
            for elem in [elem_neutral, elem_ion]:
                delta_E = self.delta_E_max.get(elem, 0.0)
                if delta_E > 0.0:
                    delta_E_values.append(delta_E)
        delta_E_max = max(delta_E_values) if delta_E_values else 4.0

        if electron_density > 5e16 and temp < 8000:
            print(f"Peringatan: n_e tinggi ({electron_density:.2e} cm^-3) dan T rendah ({temp} K) dapat menyebabkan self-absorption.")

        atom_percentages_dict = {}
        total_target_percentage = 0.0
        for base_elem, elem_neutral, elem_ion in selected_pairs:
            ion_energy = ionization_energies.get(f"{base_elem} I", 0.0)
            if ion_energy == 0.0:
                print(f"Peringatan: Tidak ada energi ionisasi untuk {base_elem} I")
                continue
            saha_ratio = self._saha_ratio(ion_energy, temp, electron_density)
            total_percentage = np.random.uniform(5, 20)
            fraction_neutral = 1 / (1 + saha_ratio)
            fraction_ion = saha_ratio / (1 + saha_ratio)
            percentage_neutral = total_percentage * fraction_neutral
            percentage_ion = total_percentage * fraction_ion
            atom_percentages_dict[elem_neutral] = percentage_neutral / 100.0
            atom_percentages_dict[elem_ion] = percentage_ion / 100.0
            total_target_percentage += total_percentage

        if total_target_percentage != 0:
            scaling_factor = 100.0 / total_target_percentage
            for key in atom_percentages_dict:
                atom_percentages_dict[key] *= scaling_factor
        else:
            return None, None, None, None

        atom_percentages_dict['temperature'] = float(temp)
        atom_percentages_dict['electron_density'] = float(electron_density)
        atom_percentages_dict['delta_E_max'] = float(delta_E_max)

        selected_target_elements = [k for k in atom_percentages_dict.keys() if k not in ['temperature', 'electron_density', 'delta_E_max']]
        atom_percentages = np.array([atom_percentages_dict[elem] for elem in selected_target_elements], dtype=np.float32)
        selected_simulators = [sim for sim in self.simulators if f"{sim.element}_{sim.ion}" in selected_target_elements]

        if not selected_simulators:
            print(f"Peringatan: Tidak ada simulator valid untuk {selected_target_elements}")
            return None, None, None, None

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

        if np.max(mixed_spectrum) == 0:
            print(f"Peringatan: Tidak ada spektrum dihasilkan untuk T={temp} K")
            return None, None, None, None

        convolved_spectrum = self._convolve_spectrum(mixed_spectrum, self.convolution_sigma)
        normalized_spectrum = self._normalize_intensity(convolved_spectrum, SIMULATION_CONFIG["target_max_intensity"])

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

        atom_percentages_dict = {
            k: float(v * 100) if k not in ['temperature', 'electron_density', 'delta_E_max'] else float(v)
            for k, v in atom_percentages_dict.items()
        }
        return self.wavelengths, normalized_spectrum, labels, atom_percentages_dict

class DatasetGenerator:
    """Menghasilkan dan menyimpan dataset spektral dengan distribusi suhu yang seragam."""
    def __init__(self, config: Dict = SIMULATION_CONFIG):
        self.config = config
        self.temperature_range = config["temperature_range"]
        self.electron_density_range = config["electron_density_range"]
        self.num_samples = config["num_samples"]
        self.combinations_json_path = None
        self.used_combinations = set()

    def _calculate_lte_electron_density(self, temp: float, delta_E: float) -> float:
        return 1.6e12 * (temp ** 0.5) * (delta_E ** 3)

    def _hash_combination(self, temp: float, n_e: float, atom_percentages: Dict) -> str:
        elements_sorted = sorted(
            [(k, round(v, 6)) for k, v in atom_percentages.items()
             if k not in ['temperature', 'electron_density', 'delta_E_max']],
            key=lambda x: x[0]
        )
        combination_str = f"{temp:.2f}_{n_e:.2e}_{str(elements_sorted)}"
        return hashlib.sha256(combination_str.encode()).hexdigest()

    def _load_used_combinations(self) -> None:
        self.used_combinations = set()
        if os.path.exists(self.combinations_json_path):
            try:
                with open(self.combinations_json_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        self.used_combinations.add(entry['hash'])
            except Exception as e:
                print(f"Error memuat kombinasi JSON: {str(e)}")

    def _save_combination(self, combination: Dict) -> None:
        if not self.combinations_json_path:
            return
        try:
            data = []
            if os.path.exists(self.combinations_json_path):
                with open(self.combinations_json_path, 'r') as f:
                    data = json.load(f)
            data.append(combination)
            with open(self.combinations_json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error menyimpan kombinasi JSON: {str(e)}")

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
                print(f"Peringatan: Tidak ada densitas elektron valid untuk T={T} K. Menggunakan seluruh rentang.")
                valid_n_e = self.electron_density_range

            max_ne_per_temp = min(len(valid_n_e), 5)
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
            print(f"Peringatan: Hanya {len(sample_params)} sampel dihasilkan, menggandakan...")
            while len(sample_params) < self.num_samples:
                idx = np.random.randint(0, len(sample_params))
                sample_params.append(sample_params[idx])
        elif len(sample_params) > self.num_samples:
            sample_params = sample_params[:self.num_samples]

        return sample_params

    def generate_dataset(
        self,
        simulators: List[SpectrumSimulator],
        delta_E_max: Dict[str, float],
        ionization_energies: Dict[str, float],
        processed_dir: str,
        drive_processed_dir: str
    ) -> None:
        np.random.seed(42)
        self.combinations_json_path = os.path.join(drive_processed_dir, "combinations.json")
        self._load_used_combinations()
        mixed_simulator = MixedSpectrumSimulator(simulators, self.config, delta_E_max)
        spectra_list, labels_list, wavelengths_list, atom_percentages_list = [], [], [], []

        sample_params = self._generate_sample_params(delta_E_max)

        temps = [param[0] for param in sample_params]
        temp_counts = Counter(temps)
        for temp, count in temp_counts.items():
            print(f"Distribusi akhir - Suhu {temp} K: {count} sampel ({count/self.num_samples*100:.2f}%)")

        for i, (T, n_e) in enumerate(tqdm(sample_params, desc="Menghasilkan dataset")):
            mixed_simulator.current_T = T
            mixed_simulator.current_n_e = n_e
            max_attempts = 5
            attempt = 0
            sample_generated = False

            while attempt < max_attempts and not sample_generated:
                result = mixed_simulator.generate_sample(ionization_energies)
                if result[0] is None:
                    attempt += 1
                    continue

                wavelengths, spectrum, labels, atom_percentages = result
                combination_hash = self._hash_combination(T, n_e, atom_percentages)
                if combination_hash in self.used_combinations:
                    print(f"Kombinasi sudah ada untuk T={T} K, n_e={n_e:.2e} cm^-3, coba ulang ({attempt + 1}/{max_attempts})")
                    attempt += 1
                    continue

                spectra_list.append(spectrum)
                labels_list.append(labels)
                wavelengths_list.append(wavelengths)
                atom_percentages_list.append(atom_percentages)

                combination = {
                    'sample_id': f"sample_{len(self.used_combinations) + 1}",
                    'hash': combination_hash,
                    'temperature': float(T),
                    'electron_density': float(n_e),
                    'elements': {k: v for k, v in atom_percentages.items()
                                if k not in ['temperature', 'electron_density', 'delta_E_max']},
                    'delta_E_max': atom_percentages['delta_E_max']
                }
                self._save_combination(combination)
                self.used_combinations.add(combination_hash)
                sample_generated = True

            if not sample_generated:
                print(f"Peringatan: Gagal menghasilkan sampel unik untuk T={T} K, n_e={n_e:.2e} cm^-3 setelah {max_attempts} percobaan")

            if (i + 1) % (self.num_samples // 10) == 0:
                current_temps = [param[0] for param in sample_params[:i + 1]]
                current_counts = Counter(current_temps)
                print(f"Progres {(i + 1)/self.num_samples*100:.1f}%:")
                for temp, count in current_counts.items():
                    print(f"Suhu {temp} K: {count} sampel ({count/(i + 1)*100:.2f}%)")

        if not spectra_list:
            raise ValueError("Tidak ada sampel valid yang dihasilkan. Periksa data NIST atau konfigurasi simulator.")

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
        drive_output_path = os.path.join(drive_processed_dir, output_filename)
        os.makedirs(drive_processed_dir, exist_ok=True)

        if os.path.exists(drive_output_path):
            backup_path = os.path.join(drive_processed_dir, f"spectral_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
            shutil.copy(drive_output_path, backup_path)
            print(f"Backup dibuat di {backup_path}")

        existing_data = {'train': {'spectra': [], 'labels': [], 'atom_percentages': []},
                        'validation': {'spectra': [], 'labels': [], 'atom_percentages': []},
                        'test': {'spectra': [], 'labels': [], 'atom_percentages': []}}
        total_existing_samples = 0

        if os.path.exists(drive_output_path):
            with h5py.File(drive_output_path, 'r') as f:
                if 'wavelengths' in f:
                    existing_wavelengths = f['wavelengths'][:]
                    if not np.array_equal(existing_wavelengths, wavelengths_array):
                        raise ValueError("Wavelengths baru tidak cocok dengan wavelengths yang sudah ada")

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

        with h5py.File(drive_output_path, 'a') as f:
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

            f.attrs['last_updated'] = datetime.now().isoformat()
            f.attrs['total_samples'] = total_samples
            f.attrs['simulation_config'] = json.dumps(self.config)

        print(f"Dataset di-append ke {drive_output_path} di sub-grup train/validation/test")
        print(f"Total sampel: {total_samples} (baru: {len(spectra_list)}, lama: {total_existing_samples})")

class DataManager:
    """Mengelola pemuatan data, validasi, dan operasi file."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data/raw/HDF5")
        self.processed_dir = os.path.join(base_dir, "data/processed")
        self.nist_source_path = os.path.join(self.data_dir, "nist_data(1).h5")
        self.nist_target_path = os.path.join(self.processed_dir, "nist_data(1).h5")
        self.atomic_data_source_path = os.path.join(self.data_dir, "atomic_data1.h5")
        self.atomic_data_target_path = os.path.join(self.processed_dir, "atomic_data1.h5")
        self.json_map_path = os.path.join(self.processed_dir, "element_map.json")

    def copy_files(self) -> None:
        """Pastikan file data tersedia di direktori kerja."""
        for source, target in tqdm([(self.nist_source_path, self.nist_target_path),
                                  (self.atomic_data_source_path, self.atomic_data_target_path)],
                                 desc="Menyalin file"):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File tidak ditemukan di {source}")
            if not os.path.exists(target):
                shutil.copy(source, target)

    def load_element_map(self) -> Dict:
        if not os.path.exists(self.json_map_path):
            raise FileNotFoundError(f"element_map.json tidak ditemukan di {self.json_map_path}")
        with open(self.json_map_path, 'r') as f:
            element_map = json.load(f)

        for elem in REQUIRED_ELEMENTS:
            if elem not in element_map or not isinstance(element_map[elem], list) or abs(sum(element_map[elem]) - 1.0) > 1e-6:
                raise ValueError(f"Peta elemen tidak valid untuk {elem}: harus berupa daftar dengan jumlah 1.0")
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
                    raise KeyError(f"Kolom yang diperlukan tidak ditemukan. Tersedia: {list(df_ionization.columns)}")

                for _, row in df_ionization.iterrows():
                    try:
                        ionization_energies[row[species_col]] = float(row[ion_energy_col])
                    except (ValueError, TypeError):
                        print(f"Peringatan: Energi ionisasi tidak valid untuk {row[species_col]}, menggunakan 0.0 eV")
                        ionization_energies[row[species_col]] = 0.0

        except Exception as e:
            print(f"Error memuat atomic_data1.h5: {str(e)}")
            raise

        for elem in REQUIRED_ELEMENTS:
            base_elem, ion = elem.split('_')
            ion_level = 'I' if ion == '1' else 'II'
            sp_name = f"{base_elem} {ion_level}"
            if sp_name not in ionization_energies:
                print(f"Peringatan: Tidak ada energi ionisasi untuk {sp_name}, menggunakan 0.0 eV")
                ionization_energies[sp_name] = 0.0

        return ionization_energies

def main():
    """Fungsi utama untuk menjalankan simulasi dan menghasilkan dataset."""
    pd.set_option('future.no_silent_downcasting', True)
    torch.set_num_threads(16)  # Sesuaikan dengan jumlah core CPU HPC

    base_dir = "/home/bwalidain/data"  # Direktori di HPC BRIN
    data_manager = DataManager(base_dir)

    # Salin file data ke direktori processed
    data_manager.copy_files()

    # Muat peta elemen dan energi ionisasi
    element_map = data_manager.load_element_map()
    ionization_energies = data_manager.load_ionization_energies()

    # Muat data NIST
    fetcher = DataFetcher(data_manager.nist_target_path)
    nist_data_dict = {}
    delta_E_max_dict = {}
    for elem in REQUIRED_ELEMENTS:
        element, ion = elem.split('_')
        data, delta_E = fetcher.get_nist_data(element, int(ion))
        nist_data_dict[elem] = data
        delta_E_max_dict[elem] = delta_E
        if not data:
            print(f"Tidak ada data NIST untuk {elem}")

    # Buat simulator
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
        raise ValueError("Tidak ada simulator valid yang dibuat. Periksa data NIST.")

    # Hasilkan dataset
    generator = DatasetGenerator(SIMULATION_CONFIG)
    generator.generate_dataset(
        simulators,
        delta_E_max_dict,
        ionization_energies,
        data_manager.processed_dir,
        data_manager.processed_dir  # Gunakan direktori processed lokal
    )

if __name__ == "__main__":
    main()
