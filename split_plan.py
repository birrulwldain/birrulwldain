# Nama file: split_plan.py
import json
import os
import math
import argparse

def split_json_plan(input_file: str, num_chunks: int):
    """
    Membaca file JSON besar berisi daftar resep dan membaginya menjadi beberapa
    file chunk yang lebih kecil.
    """
    print(f"Membaca file rencana utama dari: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            all_recipes = json.load(f)
    except FileNotFoundError:
        print(f"Error: File input tidak ditemukan di '{input_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Gagal mem-parsing file JSON. Pastikan formatnya benar.")
        return

    total_recipes = len(all_recipes)
    if total_recipes == 0:
        print("Peringatan: File JSON tidak berisi resep. Tidak ada yang perlu dibagi.")
        return

    # Hitung berapa banyak resep per file chunk
    # math.ceil memastikan semua resep terbagi habis
    chunk_size = math.ceil(total_recipes / num_chunks)
    
    print(f"Total resep: {total_recipes:,}")
    print(f"Akan dibagi menjadi {num_chunks} file, masing-masing sekitar {chunk_size:,} resep.")

    # Loop untuk membuat dan menulis setiap file chunk
    for i in range(num_chunks):
        start_index = i * chunk_size
        # Pastikan end_index tidak melebihi total resep
        end_index = min(start_index + chunk_size, total_recipes)
        
        # Ambil "jatah" untuk chunk ini menggunakan list slicing
        current_chunk_data = all_recipes[start_index:end_index]
        
        # Jika chunk terakhir kosong (karena pembulatan), jangan buat filenya
        if not current_chunk_data:
            continue

        # Buat nama file output
        # Contoh: combinations_100k.json -> part_1_of_4_combinations_100k.json
        base_name = os.path.basename(input_file)
        output_filename = f"part_{i+1}_of_{num_chunks}_{base_name}"
        
        print(f"Menulis chunk {i+1}/{num_chunks} ({len(current_chunk_data)} resep) ke file: {output_filename}")
        
        with open(output_filename, 'w') as f:
            json.dump(current_chunk_data, f, indent=2)

    print("\nProses pembagian jatah selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membagi file JSON rencana menjadi beberapa bagian (chunk).")
    parser.add_argument('--input-file', type=str, required=True, 
                        help='Path ke file JSON besar yang akan dibagi.')
    parser.add_argument('--num-chunks', type=int, required=True, 
                        help='Jumlah file bagian yang ingin dibuat (misal: 4).')
    
    args = parser.parse_args()
    
    if args.num_chunks <= 0:
        print("Error: Jumlah chunk harus lebih besar dari 0.")
    else:
        split_json_plan(args.input_file, args.num_chunks)