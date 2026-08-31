# OceanEmbed-X (SIH26066)

### Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

[![Ministry of Earth Sciences](https://img.shields.io/badge/MoES-Smart%20India%20Hackathon%202026-blue.svg)](https://www.sih.gov.in/)
[![INCOIS](https://img.shields.io/badge/INCOIS-Ocean%20Information%20Services-0077b6.svg)](https://incois.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

---

## 1. Problem Statement

| Field | Details |
|---|---|
| **Problem ID** | SIH26066 |
| **Title** | OceanEmbed -- Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations |
| **Organization** | Indian National Centre for Ocean Information Services (INCOIS), Ministry of Earth Sciences |
| **Category** | Software |
| **Theme** | Disaster Management & Marine Ecosystem Services |

**Objective**: Develop an end-to-end deep learning framework that uses 2D surface satellite observations (SST, SSS, SSH, currents, winds) to accurately reconstruct the continuous 3D subsurface ocean temperature field from 0 to 1000 meters depth across the North Indian Ocean at $0.25^\circ$ daily resolution.

**Target Region**: North Indian Ocean ($5^\circ\text{N}\text{ to }30^\circ\text{N}, 45^\circ\text{E}\text{ to }105^\circ\text{E}$) -- Arabian Sea & Bay of Bengal

**15 Standard Oceanographic Depths (m)**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`

---

## 2. Key Framework Innovations

1. **Satellite Embedding Engine (OceanMamba SSM)**:
   - Uses 2D Selective State-Space Models (SS2D) with **linear $\mathcal{O}(N)$ complexity** to compress 7-channel surface observations into a 128-dimensional latent state-space vector ($z \in \mathbb{R}^{128}$).
   - Captures invisible mesoscale eddy vorticity, planetary Rossby wave phase, and vertical advection memory without quadratic attention overhead.

2. **Analytical Sturm-Liouville Normal Mode Synthesis**:
   - Solves the vertical eigenvalue problem for stratified ocean dynamics, predicting 5 baroclinic mode weights rather than 15 unconstrained depth layers.
   - Guarantees **strictly zero buoyancy inversions ($N^2(z) \ge 0$)** by construction.

3. **In-Situ Neural 4D-Var Float Cross-Attention**:
   - Ingests today's live sparse Argo float soundings as prompt tokens, achieving near-zero error at active float positions while propagating corrections across surrounding eddies.

4. **INCOIS Operational & Ecosystem Innovation Suite**:
   - **Tropical Cyclone Heat Potential (TCHP & $D_{26}$)**: Early warning for Cyclone Rapid Intensification in Arabian Sea / Bay of Bengal.
   - **Potential Fishing Zone (PFZ) & $D_{20}$ Upwelling**: Identifies the $20^\circ\text{C}$ isotherm depth and thermocline slope to power INCOIS's daily coastal fisheries advisory ($>500,000$ fishermen).
   - **INCOIS OOSA: Oil Spill & Microplastic Dispersion**: Models vertical droplet mixing depth ($z_{\text{droplet}}$) and microplastic submergence zone ($0\text{--}150\text{m}$) driven by Mixed Layer Depth ($\text{MLD}$) and $N^2(z)$ stratification.
   - **Intelligent Active Sampling Float Drop Optimizer**: Analyzes model quantile variance to recommend optimal GPS deployment coordinates for MoES research vessels (*ORV Sagar Nidhi*).

5. **15-Depth Verification Benchmark Scorecard**:
   - Automated evaluation across all 15 standard depths reporting RMSE, MAE, Bias, and Pearson Correlation ($r$) against independent ARGO in-situ soundings.

---

## 3. Directory Structure

```
ordinary/
├── ARCHITECTURE.md                  # Master technical specifications & mathematical formulations
├── README.md                        # Project quickstart & dataset specifications
├── run_pipeline.py                  # Single-command end-to-end operational runner
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Clean git ignore rules
│
├── config/
│   ├── data_config.yaml             # 5°N-30°N, 45°E-105°E bounding box & 15 depths
│   └── model_config.yaml            # HyperOcean-Mamba hyperparameters
│
├── outputs/                         # Exported embeddings & benchmark scorecards
│   ├── satellite_embeddings.npz     # 128-dimensional latent embeddings
│   └── incois_15depth_benchmark_scorecard.csv # Depth-stratified evaluation table
│
├── src/
│   ├── data/                        # Ingestion, Harmonization & Physics Solvers
│   │   ├── glorys_loader.py         # GLORYS12V1 reanalysis target loader
│   │   ├── satellite_fetcher.py     # OSTIA SST, SMAP SSS, DUACS SSH, OSCAR, CCMP
│   │   ├── incois_argo_pipeline.py  # INCOIS LAS Gridded ARGO validation data fetcher
│   │   ├── argo_pipeline.py         # argopy GDAC live float collector
│   │   ├── sturm_liouville.py       # Analytical Baroclinic Normal Mode solver
│   │   └── mock_generator.py        # Offline synthetic generator (instant dev)
│   │
│   ├── models/                      # Deep Learning Architectures
│   │   ├── ocean_mamba.py           # 2D Selective State-Space Embedding Engine
│   │   ├── in_situ_prompting.py     # Neural 4D-Var Float Cross-Attention Block
│   │   ├── latent_embedder.py       # Satellite Latent Embedding Exporter & Stats
│   │   ├── physics_loss.py          # Buoyancy stability & depth-weighted loss
│   │   └── hybrid_reconstructor.py  # Master Reconstructor: T(z) = T_clim + Σ a_m Φ_m
│   │
│   ├── evaluation/                  # Metrics & Benchmarking
│   │   ├── metrics.py               # Depth-wise RMSE, MAE, Bias, Pearson R, Skill Score
│   │   └── benchmark_report.py      # 15-Depth Scorecard & CSV Exporter
│   │
│   ├── domain/                      # INCOIS Operational & Ecosystem Services
│   │   ├── cyclone_tchp.py          # Cyclone Rapid Intensification (TCHP & D26)
│   │   ├── pfz_upwelling.py         # Potential Fishing Zone & D20 Upwelling
│   │   ├── oil_and_plastic.py       # INCOIS OOSA Oil Spill & Plastic Dispersion
│   │   └── active_sampling.py       # Optimal Argo Float Drop Recommender
│   │
│   └── api/                         # Operational REST API Server
│       ├── main.py                  # FastAPI high-speed endpoints (<10ms)
│       └── schemas.py               # Pydantic request/response models
│
├── web/                             # Web GIS Digital Twin Dashboard
│   ├── index.html                   # 5-Tab Interface (Profile, Embedding, TCHP, PFZ, Scorecard)
│   └── src/
│       ├── app.js                   # Interactive Leaflet & Plotly visualization engine
│       └── style.css
│
├── notebooks/
│   └── OceanEmbed_X_Colab_Training.ipynb  # Clean zero-emoji training notebook
│
└── tests/
    └── test_pipeline.py             # 9-suite automated verification tests
```

---

## 4. Quickstart & Usage

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/cyber-atharv/ordinary.git
cd ordinary

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (9 Test Suites)
```bash
python tests/test_pipeline.py
```

### 3. Run Master Operational Pipeline Demo
```bash
python run_pipeline.py --mode demo
```
*Executes all 6 stages in ~1.5s, exporting `outputs/satellite_embeddings.npz` and `outputs/incois_15depth_benchmark_scorecard.csv`.*

### 4. Launch Operational FastAPI Server & Web GIS Dashboard
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Web GIS Digital Twin**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Schema Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 5. Input and Target Datasets (Per INCOIS Guidelines)

| Variable | Recommended Product | Native Resolution | Source / Citation |
|---|---|---|---|
| **SST** | OSTIA | $0.05^\circ$, daily | [doi:10.48670/moi-00168](https://doi.org/10.48670/moi-00168) |
| **SSS** | SMAP / SMOS | $0.125^\circ$, daily | [doi:10.48670/moi-00051](https://doi.org/10.48670/moi-00051) |
| **SSH / SLA** | DUACS | $0.25^\circ$, daily | [doi:10.48670/moi-00145](https://doi.org/10.48670/moi-00145) |
| **Currents (U, V)** | OSCAR L4 v2.0 | $0.25^\circ$, daily | [PODAAC OSCAR](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0) |
| **Winds (U, V)** | CCMP v3.1 / ASCAT | $0.25^\circ$, 6-hourly | [PODAAC CCMP](https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1) |
| **Target: Subsurface Temp** | GLORYS12V1 | $0.083^\circ$, daily | [doi:10.48670/moi-00021](https://doi.org/10.48670/moi-00021) |
| **Validation: In-Situ** | Gridded ARGO | Point / Gridded | [INCOIS Live Access Server (LAS)](https://incois.gov.in) |

All datasets are regridded to $0.25^\circ \times 0.25^\circ$ daily uniform resolution.

---

## 6. License

Developed for the Smart India Hackathon 2026 under the Indian National Centre for Ocean Information Services (INCOIS) / Ministry of Earth Sciences (MoES) problem statement.
