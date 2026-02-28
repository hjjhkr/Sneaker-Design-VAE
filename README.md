# Sneaker-Design-VAE

New Sneaker Designs Generation from Existing Design Data using a Beta-VAE workflow.

## Project Overview

This repository contains:
- A processed sneaker image dataset (`processed_sneakers_64x64/`).
- A Beta-VAE training and generation pipeline.
- A modular notebook workflow split into five focused stages.

The original all-in-one notebook (`BayesML_Final_Project.ipynb`) is still kept for reference.  
The recommended workflow is the new modular notebook sequence under `notebooks/`.

## Repository Structure

```text
Sneaker-Design-VAE/
├── BayesML_Final_Project.ipynb                  # Original monolithic notebook (legacy)
├── README.md
├── environment.yml                              # Conda environment definition
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_improved_training_with_annealing.ipynb
│   ├── 04_latent_analysis_rich_visualization.ipynb
│   └── 05_custom_design_interactive.ipynb
└── processed_sneakers_64x64/                    # Processed 64x64 sneaker images
```

## Environment Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate sneaker-design-vae
```

Register the kernel (optional but recommended):

```bash
python -m ipykernel install --user --name sneaker-design-vae --display-name "Python (sneaker-design-vae)"
```

Quick import check:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

To reduce warning noise in notebooks, this environment also sets:
- `PYTHONWARNINGS=ignore`
- `KMP_WARNINGS=0`
- `MKL_VERBOSE=0`

Note: `Intel MKL WARNING` messages are emitted by native libraries, so they are not always suppressed by Python's `warnings.filterwarnings`.

## Notebook Workflow (Run in Order)

1. `notebooks/01_data_preprocessing.ipynb`  
   Converts raw sneaker images to RGB, square white-background format, then resizes to `64x64`.

2. `notebooks/02_baseline_training.ipynb`  
   Trains baseline Beta-VAE with MSE reconstruction loss and fixed beta (`beta=4.0`).

3. `notebooks/03_improved_training_with_annealing.ipynb`  
   Trains improved Beta-VAE with BCE reconstruction and KL annealing schedule.

4. `notebooks/04_latent_analysis_rich_visualization.ipynb`  
   Loads trained checkpoint and performs richer latent analysis, including reconstruction grids, traversal, latent histograms, latent correlation heatmap, KL-per-dimension contributions, decoder sensitivity bars, random prior sampling, and interpolation.

5. `notebooks/05_custom_design_interactive.ipynb`  
   Provides an interactive custom design panel with synced sliders + numeric inputs for `Dim0` to `Dim15` (or up to model latent dimension), then decodes a new sneaker design on button click.

## Generated Artifacts

During notebook runs, outputs are written to:
- `artifacts/checkpoints/beta_vae_baseline.pt`
- `artifacts/checkpoints/beta_vae_improved.pt`
- `artifacts/generated/custom_sneaker.png`
- `artifacts/generated/custom_sneaker_interactive.png`

## HTML Interactive Designer

You can run a browser-based interactive designer (same core behavior as notebook `05`) with:

```bash
python web/design_server.py --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The page provides:
- Sliders + numeric inputs for `Dim0` to `Dim15` (or up to model latent dimension)
- `Generate Design`, `Reset`, `Randomize`, and `Download PNG` buttons
- Model-backed image generation through `/api/generate`

## Notes

- The preprocessing notebook requires access to the original raw dataset path.  
  Update `RAW_DATA_DIR` in `01_data_preprocessing.ipynb` before running.
- Device selection supports Apple MPS, CUDA, and CPU (automatic fallback).
- If you only need generation, ensure at least one trained checkpoint exists (`baseline` or `improved`).
- If `torchvision` is missing after environment creation, run:
  `conda install -n sneaker-design-vae -c pytorch torchvision=0.17.2`
