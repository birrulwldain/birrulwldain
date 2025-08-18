# Project Structure

This repository has been organized for clarity and maintainability. Code, data, models, notebooks, and reports are grouped into dedicated folders, with backward-compatible symlinks left at the root for commonly referenced files.

## Layout

- src/ — Python source files and entrypoints
- scripts/ — Shell scripts and CLI helpers
- notebooks/ — Jupyter notebooks for exploration
- data/
  - raw/ — Original datasets (immutable)
  - processed/ — Derived/processed datasets
  - config/ — JSON configs, mappings, and recipe files
- models/ — Model weights and artifacts
  - archive/ — Older/duplicate model files
- reports/
  - latex/ — LaTeX sources and PDFs
- deploy/
  - vertex-ai/ — Vertex AI artifacts (Dockerfile, requirements, train script)
- docs/ — PDFs and documentation

Root-level symlinks may exist to preserve old paths used by scripts. New work should target the structured paths above.

## Common Paths

- Datasets: data/raw/*.h5, data/processed/*.h5
- Configs: data/config/*.json
- Models: models/*.pth
- Entrypoints: src/*.py or scripts/*.sh

## Notes

- Large binary files (datasets, model weights) are ignored by Git by default. Commit only small configs and code.
- If you need to regenerate symlinks after moving files, re-run tools/restructure.sh.

## Alur Kerja di HPC (SLURM) — Planner → Worker → Merge

Workflow yang digunakan:
- Buat rencana kombinasi (planner.py)
- Jalankan worker per-chunk (job.sh → job.py)
- Gabungkan hasil chunk (job-merge.sh → merge.py)

Asumsi
- Direktori kerja HPC: /home/bwalidain/birrulwldain (sesuai isi skrip).
- Conda environment: bw berisi dependensi (torch, numpy, scipy, h5py, pandas, tqdm, dll.).
- File pendukung tersedia: nist_data(1).h5, atomic_data1.h5, element-map-35.json, dll.

1) Buat rencana kombinasi dengan planner.py (di login node)
- Output utama: combinations-{SUFFIX}.json
- Jika di-split: combinations-{SUFFIX}-{CHUNK}.json

Contoh:
```zsh
python src/planner.py --num 10000 --split 20
# Hasil: combinations-10.json dan combinations-10-1.json .. combinations-10-20.json
```

2) Jalankan worker per-chunk via SLURM (job.sh → job.py)
- Format: sbatch scripts/job.sh <CHUNK_ID> <FILE_SUFFIX>

Contoh:
```zsh
for i in {1..20}; do
  sbatch scripts/job.sh $i 10
done
# Output per chunk: dataset-10-<CHUNK_ID>.h5
# Log: /home/bwalidain/birrulwldain/logs/worker_manual_%j.out|err
```

3) Cek progres job
```zsh
squeue -u $USER
ls -1t ~/birrulwldain/logs | head
```

4) Gabungkan hasil chunk (job-merge.sh → src/merge.py)
- Jalankan setelah semua dataset-<SUFFIX>-*.h5 selesai.

Contoh:
```zsh
sbatch scripts/job-merge.sh 10
# Output gabungan: /home/bwalidain/birrulwldain/dataset-10.h5
```

Catatan
- Skrip SLURM memakai path absolut (/home/bwalidain/birrulwldain). Sesuaikan jika WORK_DIR berbeda.
- planner.py menggunakan NIST_HDF_PATH bawaan; ubah di file jika path berbeda.
- Struktur repo telah dipindah ke src/ dan scripts/. Jika Anda memakai file lama di root, symlink disediakan untuk kompatibilitas.
