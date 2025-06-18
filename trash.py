
# (Asumsikan semua konfigurasi dan kelas lain seperti DataFetcher, DataManager, dll.
# sudah ada di atas kode ini seperti pada skrip asli Anda)

# --- KELAS YANG DIREVISI ---



# --- FUNGSI MAIN YANG DIREVISI ---

def main(args):
    """Fungsi utama untuk menjalankan simulasi dalam mode pekerja."""
    
    job_id = os.path.basename(args.output_h5).replace('.h5', '')
    # (Asumsikan setup_logging, SIMULATION_CONFIG, dan kelas DataManager/DataFetcher ada)
    logger = setup_logging(base_dir=SIMULATION_CONFIG["logs_dir"], job_id=job_id)
    logger.info(f"Memulai PEKERJA simulasi spektral untuk file resep: {args.input_json}")

    # --- REVISI: Set thread ke 1 untuk efisiensi multiprocessing ---
    # Paralelisme datang dari jumlah PROSES (worker), bukan dari thread di dalam proses.
    # Ini mencegah perebutan sumber daya CPU dan membuat kinerja lebih efisien dan dapat diprediksi.
    torch.set_num_threads(1)

    logger.info("Memuat data pendukung (NIST, Ionization, Element Map)...")
    base_dir = SIMULATION_CONFIG["data_dir"]
    data_manager = DataManager(base_dir)
    element_map = data_manager.load_element_map()
    ionization_energies = data_manager.load_ionization_energies()
    
    fetcher = DataFetcher(data_manager.nist_target_path)
    nist_data_dict = {}
    delta_E_max_dict = {}
    
    # (Asumsikan REQUIRED_ELEMENTS sudah didefinisikan)
    for elem in tqdm(REQUIRED_ELEMENTS, desc="Fetching NIST Data"):
        element, ion = elem.split('_')
        data, delta_E = fetcher.get_nist_data(element, int(ion))
        nist_data_dict[elem] = data
        delta_E_max_dict[elem] = delta_E

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
                ionization_energy=ion_energy, 
                config=SIMULATION_CONFIG, 
                element_map_labels=element_map
            )
            simulators.append(simulator)
    logger.info(f"Total {len(simulators)} simulator berhasil dibuat.")

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

    # (Asumsikan kelas WorkerDatasetGenerator ada dan benar)
    generator = WorkerDatasetGenerator(SIMULATION_CONFIG, element_map)
    generator.process_recipes(
        recipes=recipes, 
        simulators=simulators, 
        ionization_energies=ionization_energies, 
        delta_E_max_dict=delta_E_max_dict, 
        output_h5_path=args.output_h5
    )

    logger.info("Proses pekerja selesai.")

# (Asumsikan blok if __name__ == "__main__": ada untuk memanggil main(parsed_args))