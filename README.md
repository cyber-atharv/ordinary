# OceanEmbed-X (SIH26066)
### HyperOcean-Mamba: Continuous State-Space Baroclinic Normal-Mode Framework for 3D Subsurface Ocean Temperature Reconstruction

[![Ministry of Earth Sciences](https://img.shields.io/badge/MoES-Smart%20India%20Hackathon%202026-blue.svg)](https://www.sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Architecture Specs](https://img.shields.io/badge/Specs-ARCHITECTURE.md-orange.svg)](ARCHITECTURE.md)

---

## 1. Project Overview & Problem Statement

**Problem ID**: SIH26066 
**Problem Title**: *OceanEmbed - Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations* 
**Organization**: Ministry of Earth Sciences (MoES) · Software Track 
**Target Domain**: North Indian Ocean ($5^\circ\text{N}\text{--}30^\circ\text{N}, 45^\circ\text{E}\text{--}105^\circ\text{E}$ — Arabian Sea & Bay of Bengal) 
**Standard Grid**: $0.25^\circ \times 0.25^\circ$ Daily Temporal Resolution 
**15 Standard Depths (m)**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`

---

## 2. Dataset Strategy & Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ MULTI-SOURCE DATASET FUSION STRATEGY │
├───────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ 1. Surface Satellite Inputs │ • Copernicus Marine / OSTIA (SST, SSS, SLA, Winds, Currents)│
│ │ • Kaggle NASA Ocean Climate (`brsdincer/ocean-data-climate`)│
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Subsurface Ground Truth │ • In-Situ Argo Floats (0–1000m via `argopy` GDAC API) │
│ │ • GLORYS12V1 / INCOIS Gridded ARGO 3D Reanalysis │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Land-Sea & Water Masks │ • Kaggle Water Bodies Sentinel-2 (`franciscoescobar/...`) │
│ │ • GEBCO Bathymetric Grids (Bottom depth constraints) │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Marine Ecosystem Impact │ • Kaggle Shifting Seas (`atharvasoundankar/shifting-seas`) │
│ │ • Coral Bleaching & Subsurface Marine Heatwave (SMHW) Track │
├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. Instant Offline Dev Mode │ • High-Fidelity Physical Synthetic Generator (`src/data/`) │
│ │ (Zero-download instant training with realistic dynamics) │
└───────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

> **Why Kaggle data alone is not enough**: The Kaggle datasets provide surface measurements, optical RGB masks, and ecological labels, but lack the **3D subsurface vertical ground truth ($0\text{--}1000\text{m}$)** required by MoES. Our framework bridges this by fusing the Kaggle surface/mask data with **`argopy` GDAC in-situ casts**, **Copernicus Marine altimetry/salinity**, and an **offline physical generator**.

---

## 3. Master System Architecture

```mermaid
flowchart TB
 subgraph IngestionLayer ["1. Multi-Modal Data Ingestion Layer"]
 KAGGLE_SURF["Kaggle Surface NetCDF + Sentinel-2 Masks"]
 CMEMS["Copernicus Marine API (SST, SSS, SLA, U/V Currents, Winds)"]
 ARGO["In-Situ Argo GDAC (0-1000m Profiles via argopy)"]
 KAGGLE_ECO["Kaggle Marine Ecology & Coral Impact Data"]
 end

 subgraph PreprocessingEngine ["2. Harmonization & Baroclinic Decomposition"]
 REGRID["CDO/xarray Spatial Harmonizer (0.25° Daily Grid)"]
 ARGO_QC["Argo QC Flag Filter & PCHIP 15-Depth Binning"]
 STURM["Analytical Sturm-Liouville Baroclinic Mode Solver Φ_m(z)"]
 
 KAGGLE_SURF & CMEMS --> REGRID
 ARGO --> ARGO_QC
 ARGO_QC --> STURM
 end

 subgraph HybridBackbone ["3. HyperOcean-Mamba (HO-Mamba) Deep Engine"]
 WAVE_STEM["Semantic Wavefront & 7-Day Memory Stem"]
 MAMBA_SSM["2D Selective State-Space Model (Linear O(N) Speed)"]
 IN_SITU_PROMPT["In-Situ Neural 4D-Var Prompting Block (Live Floats)"]
 
 REGRID --> WAVE_STEM --> MAMBA_SSM
 ARGO_QC --> IN_SITU_PROMPT
 MAMBA_SSM <--> IN_SITU_PROMPT
 end

 subgraph PhysicsReconstruction ["4. Physics-Guided 3D Reconstruction"]
 CLIM["Background Climatology T_clim(x, y, z)"]
 APE_LOSS["Hamiltonian Available Potential Energy (APE) Loss"]
 MODAL_SYNTH["Modal Synthesizer: T(z) = T_clim(z) + Σ a_m · Φ_m(z)"]
 
 IN_SITU_PROMPT --> MODAL_SYNTH
 STURM --> MODAL_SYNTH
 CLIM --> MODAL_SYNTH
 MODAL_SYNTH --> APE_LOSS
 end

 subgraph TacticalTwin ["5. Operational Intelligence & Interactive Web GIS"]
 FIELD_3D["Continuous 3D Ocean Thermal Field (0-1000m)"]
 TCHP["Cyclone Rapid Intensification: TCHP & D26 Heat Pool"]
 SONAR["Naval Defense Engine: Mackenzie SVP & Sonar Ray-Tracing"]
 MHW_ECO["Marine Ecosystem: Deep Heatwave & Coral Stress"]
 FASTAPI["FastAPI Async Engine (<10ms) & CF-NetCDF Exporter"]
 WEB_GIS["3D WebGL Digital Twin Dashboard"]
 
 MODAL_SYNTH --> FIELD_3D
 FIELD_3D --> TCHP & SONAR & MHW_ECO --> FASTAPI --> WEB_GIS
 end
```

---

## 4. Why OceanEmbed-X is 100% Unique

1. **Analytical Sturm-Liouville Baroclinic Normal Modes**: Solves exact Rossby deformation eigenvalue equations, guaranteeing **zero density inversions ($N^2 \ge 0$)** without empirical PCA artifacts.
2. **Linear-Time State Space Model (Ocean-Mamba)**: Replaces heavy $O(N^2)$ Vision Transformers with $O(N)$ selective state-space scans, enabling real-time 14-day wave tracking on ordinary laptops and shipboard hardware.
3. **In-Situ Neural 4D-Var Prompting**: Ingests today's live sparse Argo floats as prompt tokens, guaranteeing near-zero error at active float positions while inferring basin-wide fields.
4. **Dual Civilian-Defense Tactical Engines**:
 - **Civilian**: Real-time **Tropical Cyclone Heat Potential (TCHP)** for cyclone rapid intensification forecasting.
 - **Naval Defense**: **Differentiable Sonar Acoustic Ray-Tracing (Mackenzie Formula)** visualizing submarine acoustic shadow zones and SOFAR sound channels.

---

## 5. Directory Structure

```
ordinary/
├── ARCHITECTURE.md # Master Technical Specifications & Equations
├── SIH26066_OceanEmbed_Working_Guide.md # Domain Guide & Data Protocols
├── README.md # Project Overview & Quickstart
├── requirements.txt # Python Dependencies
├── package.json # Workspace Meta & NPM Scripts
│
├── config/ # Data & Model Configurations
│ ├── data_config.yaml # 5°N-30°N, 45°E-105°E bounding box & 15 depths
│ └── model_config.yaml # Mamba SSM, Baroclinic mode & loss weights
│
├── data/ # Data Storage (Git-ignored)
│ ├── raw/ # Copernicus, Kaggle & Argo downloads
│ ├── processed/ # QC-filtered, PCHIP-regridded NetCDF & Zarr arrays
│ └── synthetic/ # High-fidelity offline physical test arrays
│
├── src/ # Core Python Engine
│ ├── data/ # Ingestion (Kaggle, Copernicus, Argo), QC, Harmonizers
│ │ ├── kaggle_loader.py # Ingests NASA, Sentinel-2 masks, & Shifting Seas
│ │ ├── copernicus_fetcher.py # Copernicus Marine API automated downloader
│ │ ├── argo_pipeline.py # argopy GDAC collector & QC-1 filter
│ │ ├── sturm_liouville.py # Analytical Baroclinic Normal Mode solver
│ │ └── mock_generator.py # Offline physical generator for instant dev
│ │
│ ├── models/ # Neural Architectures
│ │ ├── ocean_mamba.py # 2D Selective Scan State-Space Model (SSM)
│ │ ├── in_situ_prompting.py # Neural 4D-Var cross-attention float assimilation
│ │ ├── physics_loss.py # APE Hamiltonian & Quasi-Geostrophic thermal wind loss
│ │ └── hybrid_reconstructor.py # Full modal synthesizer T(z) = T_clim + Σ a_m Φ_m
│ │
│ ├── evaluation/ # Metrics & Out-of-Sample Float Verification
│ │ ├── metrics.py # Depth-stratified RMSE, MAE, Bias, Pearson R, Skill Score
│ │ └── argo_evaluator.py # Unseen float track validation & spatial error mapping
│ │
│ ├── domain/ # Domain & Defense Analytics
│ │ ├── cyclone_tchp.py # TCHP & D26 heat pool calculator with IMD cyclone tracks
│ │ ├── tactical_sonar.py # Mackenzie sound velocity & acoustic ray tracer
│ │ └── marine_heatwave.py # Subsurface MHW & coral bleaching stress engine
│ │
│ └── api/ # Backend REST API
│ ├── main.py # FastAPI high-speed endpoints (<10ms)
│ └── export.py # CF-compliant NetCDF4 & GeoTIFF exporter
│
├── web/ # Frontend Application (Digital Twin Dashboard)
│ ├── index.html # Main application interface
│ ├── src/ # MapLibre/Leaflet GIS, 3D WebGL slices, profile charts
│ └── public/ # Static icons & sample data previews
│
├── notebooks/ # Interactive EDA & Training Notebooks
└── tests/ # Automated Unit & Physics Consistency Tests
```

---

## 6. Quickstart & Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate # Windows ('source venv/bin/activate' on Linux/macOS)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run instant offline data setup & verification test
python -m src.data.mock_generator

# 4. Start FastAPI server & Web GIS Dashboard
uvicorn src.api.main:app --reload --port 8000
```
