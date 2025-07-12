# ===================================================================
# Dockerfile untuk Menjalankan Training PyTorch di Vertex AI
# ===================================================================

# Langkah 1: Pilih Base Image
# Kita mulai dengan image resmi Python. Vertex AI akan menyediakan lingkungan
# dengan driver CUDA yang sesuai, jadi kita tidak perlu menyertakannya di sini.
FROM python:3.11-slim

# Langkah 2: Atur Direktori Kerja di dalam Container
WORKDIR /app

# Langkah 3: Salin file requirements dan instal dependensi
# PERUBAHAN: Menambahkan --index-url untuk memastikan PyTorch versi GPU terinstal
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# Langkah 4: Salin semua kode dan data Anda ke dalam container
# Menyalin direktori 'data' dan skrip 'train_hpc.py'
COPY data/ ./data/
COPY train_hpc.py .

# Langkah 5: Definisikan Entry Point
# Ini adalah perintah yang akan dijalankan saat container dimulai oleh Vertex AI.
# Perintah ini menjalankan skrip Python Anda. Argumen seperti path file
# akan diberikan oleh Vertex AI saat job dijalankan.
ENTRYPOINT ["python", "train.py"]
