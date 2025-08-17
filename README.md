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
