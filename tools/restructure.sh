#!/usr/bin/env bash
set -euo pipefail

# Restructure repository while preserving backward compatibility with symlinks.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p src scripts notebooks data/raw data/processed data/config models/archive reports/latex deploy/vertex-ai docs tools

# Move python entrypoints to src/
for f in bin.py check.py eval.py job.py job-1.py job-backup.py map.py merge.py planner.py planner-1.py report.py sim.py sim1.py sim2.py sim2-s.py sim3.py sim4.py sim5.py simulasi.py split_plan.py train.py train-fly.py trash.py; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" src/ 2>/dev/null || mv -n "$f" src/
  fi
done

# Move shell scripts to scripts/
for f in run.sh run-b.sh run-eval.sh run-job.sh run-job1.sh run-job-9.sh run-job-r.sh run-job-r1.sh run-job-s.sh run-o.sh job.sh job-merge.sh; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" scripts/ 2>/dev/null || mv -n "$f" scripts/
    chmod +x "scripts/$(basename "$f")" || true
  fi
done

# Move notebooks
for f in *.ipynb; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" notebooks/ 2>/dev/null || mv -n "$f" notebooks/
  fi
done

# Move datasets
for f in *.h5; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" data/raw/ 2>/dev/null || mv -n "$f" data/raw/
  fi
done

# Move JSON configs and mappings
for f in *.json; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" data/config/ 2>/dev/null || mv -n "$f" data/config/
  fi
done

# Move models
for f in *.pth; do
  if [[ -f "$f" ]]; then
    base=$(basename "$f")
    if [[ "$base" == Copy* ]]; then
      git mv -k "$f" models/archive/ 2>/dev/null || mv -n "$f" models/archive/
    else
      git mv -k "$f" models/ 2>/dev/null || mv -n "$f" models/
    fi
  fi
done

# Move PDFs and docs
for f in *.pdf; do
  if [[ -f "$f" ]]; then
    git mv -k "$f" docs/ 2>/dev/null || mv -n "$f" docs/
  fi
done

# Move LaTeX projects (prefer fast renames over copying)
if [[ -d pylatex_test ]]; then
  mkdir -p reports/latex
  git mv -k pylatex_test reports/latex/ 2>/dev/null || mv -n pylatex_test reports/latex/
fi
if [[ -d spectral_report_v12 ]]; then
  mkdir -p reports/latex
  git mv -k spectral_report_v12 reports/latex/ 2>/dev/null || mv -n spectral_report_v12 reports/latex/
fi

# Move Vertex AI folder
if [[ -d vertex-ai ]]; then
  mkdir -p deploy
  git mv -k vertex-ai deploy/ 2>/dev/null || mv -n vertex-ai deploy/
fi

# Create backward-compatible symlinks in root for frequently used files
symlink_if_missing() {
  local src="$1"; shift
  local link="$1"; shift
  if [[ -e "$src" && ! -e "$link" ]]; then
    ln -s "$src" "$link"
  fi
}

# Common files referenced by old scripts
for f in \
  data/raw/dataset-10.h5 \
  data/raw/dataset-11.h5 \
  data/raw/dataset-20.h5 \
  data/raw/dataset-50.h5 \
  data/raw/dataset100k.h5 \
  data/raw/dataset2k.h5 \
  data/raw/spectral_dataset-35.h5 \
  data/raw/spectral_dataset-35-1.h5 \
  data/raw/spectral_dataset-35_s.h5 \
  data/raw/spectral_dataset-35-stratified.h5 \
  data/raw/spectral_dataset-9.h5 \
  data/config/element-map.json \
  data/config/element_map.json \
  data/config/element-map-9.json \
  data/config/element-map-18a.json \
  data/config/element-map-35.json \
  data/config/element-map-237.json \
  data/config/full-element-map-35.json \
  data/config/full-element-map-237.json \
  data/config/modified-element-map.json \
  data/config/wavelengths_grid.json \
  data/config/combinations-35.json \
  data/config/test_recipes.json \
  data/config/validation_recipes.json \
  models/class_weights_multilabel.pth \
  models/informer_multilabel_model.pth; do
  base=$(basename "$f")
  symlink_if_missing "$f" "$base"
done

# Permissions
chmod -R go-w data models || true

echo "Restructure complete. Review symlinks and update scripts to use new paths (src/, data/, models/, scripts/)."
