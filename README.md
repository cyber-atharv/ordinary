# OceanEmbed-X (SIH26066)

### Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

[![Ministry of Earth Sciences](https://img.shields.io/badge/MoES-Smart%20India%20Hackathon%202026-blue.svg)](https://www.sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

---

## 1. Problem Statement

| Field | Details |
|-------|---------|
| **Problem ID** | SIH26066 |
| **Title** | OceanEmbed -- Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations |
| **Organization** | Indian National Centre for Ocean Information Services (INCOIS), Ministry of Earth Sciences |
| **Category** | Software |
| **Theme** | Disaster Management |

**Objective**: Develop a deep learning framework that uses surface satellite observations to reconstruct 3D subsurface ocean temperature profiles from 0 to 1000 meters depth across the North Indian Ocean at 0.25 deg daily resolution.

**Target Region**: North Indian Ocean (5 deg N to 30 deg N, 45 deg E to 105 deg E) -- Arabian Sea and Bay of Bengal

**Standard Depth Levels (m)**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000

---

## 2. Approach Overview

Our framework reconstructs subsurface temperature through four stages:

1. **Preprocessing**: Harmonize multi-source satellite products (SST, SSS, SSH, currents, winds) onto a unified 0.25 deg daily grid using xarray and CDO.

2. **Satellite Embedding**: Compress 7-channel surface observations into a compact latent representation using a 2D Selective State-Space Model (Mamba SSM) with linear O(N) complexity. The embedding captures mesoscale eddy signatures, thermocline displacement patterns, and wind-driven mixing dynamics that are invisible in raw pixels.

3. **Physics-Guided Reconstruction**: Instead of regressing 15 independent depth levels, predict 5 baroclinic normal mode amplitude coefficients. The final temperature profile is synthesized as:
   ```
   T(x, y, z) = T_clim(z) + sum a_m(x,y) * Phi_m(z)
   ```
   This enforces gravitational stability (N^2 >= 0) by construction and respects known ocean vertical structure.

4. **Validation**: Evaluate against independent ARGO float observations using depth-stratified RMSE, Bias, and Pearson Correlation across all 15 standard depth levels.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical specifications.

---

## 3. Input and Target Datasets

All datasets are specified by the INCOIS problem statement and regridded to 0.25 deg x 0.25 deg daily resolution.

### Surface Input Variables

| Variable | Product | Native Resolution | DOI / Source |
|----------|---------|-------------------|-------------|
| SST | OSTIA | 0.05 deg, daily | https://doi.org/10.48670/moi-00168 |
| SSS | SMAP / SMOS | 0.125 deg, daily | https://doi.org/10.48670/moi-00051 |
| SSH / SLA | DUACS | 0.25 deg, daily | https://doi.org/10.48670/moi-00145 |
| Currents (U, V) | OSCAR L4 v2.0 | 0.25 deg, daily | PODAAC |
| Winds (U, V) | CCMP v3.1 / ASCAT | 0.25 deg, 6-hourly | PODAAC |

### Training Target

| Variable | Product | Details |
|----------|---------|---------|
| Subsurface Temperature | GLORYS12V1 Global Ocean Reanalysis | https://doi.org/10.48670/moi-00021 |

### Validation

| Source | Details |
|--------|---------|
| Gridded ARGO | INCOIS Live Access Server (LAS) |
| Individual Argo Floats | Via argopy GDAC API |

---

## 4. Directory Structure

```
ordinary/
|-- ARCHITECTURE.md          # Technical architecture and design decisions
|-- README.md                # This file
|-- requirements.txt         # Python dependencies
|-- .gitignore               # Standard Python/data gitignore rules
|
|-- config/
|   |-- data_config.yaml     # Bounding box, depth levels, dataset paths
|   +-- model_config.yaml    # Model hyperparameters and loss weights
|
|-- data/                    # Data storage (git-ignored)
|   |-- raw/                 # Downloaded satellite and reanalysis NetCDFs
|   |-- processed/           # Regridded 0.25 deg daily arrays
|   +-- synthetic/           # Offline physical test arrays
|
|-- src/                     # Core Python modules
|   |-- data/                # Data ingestion, preprocessing, and physics
|   |   |-- glorys_loader.py         # GLORYS12V1 training target downloader
|   |   |-- satellite_fetcher.py     # OSTIA, SMAP, DUACS, OSCAR, CCMP fetcher
|   |   |-- incois_argo_pipeline.py  # INCOIS LAS gridded ARGO validation data
|   |   |-- argo_pipeline.py         # argopy GDAC float profile retrieval
|   |   |-- sturm_liouville.py       # Baroclinic normal mode solver
|   |   +-- mock_generator.py        # Synthetic data generator for offline dev
|   |
|   |-- models/              # Neural network architectures
|   |   |-- ocean_mamba.py           # 2D Selective State-Space embedding backbone
|   |   |-- in_situ_prompting.py     # Neural 4D-Var cross-attention float assimilation
|   |   |-- latent_embedder.py       # Satellite embedding exporter and visualizer
|   |   |-- physics_loss.py          # Physics-constrained loss with stability penalty
|   |   +-- hybrid_reconstructor.py  # Full pipeline: embedding -> modes -> 3D field
|   |
|   |-- evaluation/          # Metrics and benchmarking
|   |   |-- metrics.py               # Depth-stratified RMSE, MAE, Bias, Correlation
|   |   +-- benchmark_report.py      # 15-depth formatted evaluation scorecard
|   |
|   |-- domain/              # Downstream INCOIS applications
|   |   +-- cyclone_tchp.py          # Tropical Cyclone Heat Potential (TCHP & D26)
|   |
|   +-- api/                 # REST API backend
|       |-- main.py                  # FastAPI endpoints
|       +-- schemas.py               # Pydantic request/response models
|
|-- web/                     # Frontend web application
|   |-- index.html           # Interactive ocean digital twin dashboard
|   +-- src/                 # JavaScript, CSS, Leaflet/Plotly visualization
|
|-- notebooks/               # Jupyter/Colab training notebooks
|   +-- OceanEmbed_X_Colab_Training.ipynb
|
+-- tests/                   # Automated verification tests
    +-- test_pipeline.py
```

---

## 5. Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/cyber-atharv/ordinary.git
cd ordinary

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run verification tests (uses synthetic data, no downloads needed)
python tests/test_pipeline.py

# 5. Start the FastAPI server and web dashboard
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open http://localhost:8000 for the interactive ocean dashboard, or http://localhost:8000/docs for the Swagger API documentation.

---

## 6. Key Technical Highlights

- **Satellite Embedding Engine**: 2D Mamba SSM with O(N) linear complexity produces compact latent representations of surface ocean state. Embeddings are exportable for visualization and evaluation.

- **Physics-Constrained Reconstruction**: Sturm-Liouville baroclinic normal mode decomposition guarantees physically consistent vertical profiles (no density inversions).

- **In-Situ Assimilation**: Cross-attention mechanism fuses live Argo float observations with satellite embeddings, achieving near-zero error at float positions while propagating corrections basin-wide.

- **Uncertainty Quantification**: Multi-quantile prediction heads provide 10th, 50th, and 90th percentile confidence bounds for risk-aware downstream applications.

- **Depth-Stratified Benchmarking**: Automated evaluation across all 15 standard depth levels with RMSE, Bias, and Correlation metrics per INCOIS requirements.

---

## 7. Expected Output

The framework produces:
- Daily 3D subsurface temperature reconstruction at 0.25 deg x 0.25 deg across 15 standard depths (0 to 1000m)
- Exportable satellite latent embeddings for analysis
- Depth-stratified evaluation scorecard (RMSE, Bias, Correlation per depth layer)
- Interactive proof-of-concept visualization over the Bay of Bengal and Arabian Sea

---

## 8. References

- Copernicus Marine GLORYS12V1: https://doi.org/10.48670/moi-00021
- OSTIA SST: https://doi.org/10.48670/moi-00168
- SMAP/SMOS SSS: https://doi.org/10.48670/moi-00051
- DUACS SSH/SLA: https://doi.org/10.48670/moi-00145
- OSCAR L4 Currents: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0
- CCMP Winds: https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1
- ASCAT Coastal Winds: https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal
- INCOIS: https://incois.gov.in
- Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

---

## License

This project is developed for the Smart India Hackathon 2026 under INCOIS / Ministry of Earth Sciences guidelines.
