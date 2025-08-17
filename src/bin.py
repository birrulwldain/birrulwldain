import json

# Definisi elemen target
# versi 35
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

#versi 9
# BASE_ELEMENTS = ["Al", "Fe", "Ca", "Mg"]
# REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

# Fungsi untuk membuat elemen map
def create_element_map():
    element_map = {}
    
    # Tambahkan background dengan indeks 0
    element_map["background"] = [1] + [0] * len(REQUIRED_ELEMENTS)  # Panjang 237 (1 untuk background + 236 untuk elemen)
    
    # Tambahkan setiap elemen dengan status dari REQUIRED_ELEMENTS
    for index, element_key in enumerate(REQUIRED_ELEMENTS, start=1):  # Mulai dari indeks 1
        element_map[element_key] = [0] * (len(REQUIRED_ELEMENTS) + 1)  # Panjang 237
        element_map[element_key][index] = 1  # Set 1 pada indeks yang sesuai
    
    return element_map

# Fungsi untuk menyimpan data ke file JSON
def save_json_file(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_path}")

# Main program
def main():
    output_file = "element-map-9.json"
    
    # Buat elemen map
    data = create_element_map()
    
    # Simpan ke file JSON
    save_json_file(data, output_file)

if __name__ == "__main__":
    main()