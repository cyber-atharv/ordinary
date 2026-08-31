# OceanEmbed-X: System Architecture (SIH26066)

> **Problem Statement**: SIH26066 -- Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations
> **Organization**: Indian National Centre for Ocean Information Services (INCOIS), Ministry of Earth Sciences (MoES)
> **Category**: Software | **Theme**: Disaster Management
> **Target Region**: North Indian Ocean (5 deg N to 30 deg N, 45 deg E to 105 deg E)
> **Grid**: 0.25 deg x 0.25 deg, Daily Temporal Resolution
> **Standard Depths (m)**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000

---

## 1. Problem Context

Subsurface ocean temperature (0--1000m) is essential for monitoring upper-ocean heat content, ocean circulation, marine heatwaves, and climate variability. Direct measurements from Argo floats, moorings, and gliders remain spatially sparse and temporally intermittent. Surface satellite observations (SST, SSS, SSH, currents, winds) provide continuous basin-scale coverage but only capture 2D surface conditions.

The core challenge is to learn the nonlinear physical mapping from surface satellite observations to the 3D subsurface thermal structure using deep learning and satellite embeddings.

---

## 2. End-to-End Pipeline Architecture

```
Surface Satellite Observations (7 Channels, 0.25 deg Daily Grid)
  SST (OSTIA) | SSS (SMAP/SMOS) | SSH/SLA (DUACS) | Currents U,V (OSCAR) | Winds U,V (CCMP/ASCAT)
        |
        v
 +-------------------------------------------------+
 | Step 1: Preprocessing & Harmonization Pipeline   |
 | (src/data/)                                      |
 |  - Multi-source regridding to 0.25 deg daily     |
 |  - CDO / xarray spatial interpolation            |
 |  - Land-sea masking via NetCDF coordinate grids  |
 +-------------------------------------------------+
        |
        v
 +-------------------------------------------------+
 | Step 2: Satellite Embedding Engine               |
 | (src/models/ocean_mamba.py)                      |
 |  - 2D Selective State-Space Model (Mamba SSM)    |
 |  - Linear O(N) bi-directional spatial scan       |
 |  - Outputs: Latent embedding z in R^d            |
 |  - Embedding export for visualization/inspection |
 +-------------------------------------------------+
        |                         |
        v                         v
 +-------------------------+  +-------------------------------+
 | Step 3a: Baroclinic     |  | Step 3b: Neural 4D-Var        |
 | Mode Synthesis          |  | In-Situ Float Assimilation    |
 | (src/data/              |  | (src/models/                  |
 |  sturm_liouville.py)    |  |  in_situ_prompting.py)        |
 |                         |  |                               |
 | Modal decomposition:    |  | Cross-attention over live     |
 | T(x,y,z) = T_clim(z)   |  | Argo float casts to calibrate |
 |  + sum a_m * Phi_m(z)   |  | satellite embeddings against  |
 |                         |  | real subsurface observations  |
 +-------------------------+  +-------------------------------+
        |                         |
        +--------+  +-------------+
                 |  |
                 v  v
 +-------------------------------------------------+
 | Step 4: 3D Thermal Field Reconstruction          |
 | (src/models/hybrid_reconstructor.py)             |
 |  - Physics-guided vertical projection            |
 |  - Buoyancy stability enforcement (N^2 >= 0)    |
 |  - Multi-quantile uncertainty bounds (10/50/90%) |
 |  - Output: T(x, y, z, t) at 15 standard depths  |
 +-------------------------------------------------+
        |
        v
 +-------------------------------------------------+
 | Step 5: Validation & Benchmarking                |
 | (src/evaluation/)                                |
 |  - Depth-stratified RMSE, Bias, Correlation     |
 |  - Independent ARGO float verification           |
 |  - 15-depth benchmark scorecard generation       |
 +-------------------------------------------------+
        |
        v
 +-------------------------------------------------+
 | Step 6: Operational Serving & Visualization      |
 | (src/api/ + web/)                                |
 |  - FastAPI REST endpoints                        |
 |  - Interactive WebGL ocean digital twin          |
 |  - CF-NetCDF & GeoJSON export                    |
 |  - Downstream: TCHP/D26 cyclone heat monitoring  |
 +-------------------------------------------------+
```

---

## 3. Module Details

### 3.1. Data Ingestion and Preprocessing (src/data/)

| Module | Purpose |
|--------|---------|
| `glorys_loader.py` | Fetches GLORYS12V1 reanalysis temperature fields (training target) from Copernicus Marine |
| `satellite_fetcher.py` | Downloads and regrids OSTIA SST, SMAP/SMOS SSS, DUACS SSH, OSCAR currents, CCMP/ASCAT winds |
| `incois_argo_pipeline.py` | Fetches Gridded ARGO data from INCOIS Live Access Server (LAS) for validation |
| `argo_pipeline.py` | Retrieves individual Argo float profiles via argopy for real-time in-situ prompting |
| `sturm_liouville.py` | Solves the vertical Sturm-Liouville eigenvalue problem for baroclinic normal modes |
| `mock_generator.py` | Generates synthetic North Indian Ocean arrays for offline development and testing |

### 3.2. Deep Learning Architecture (src/models/)

**Surface Embedding Backbone (ocean_mamba.py)**

The embedding engine uses a 2D Selective State-Space Model (Mamba SSM) that scans satellite surface grids with linear computational complexity O(N). Unlike Vision Transformers that scale quadratically O(N^2) with grid resolution, the SSM processes the 0.25 deg grid efficiently even at full basin resolution.

The backbone produces a compact latent embedding vector z that captures invisible subsurface signatures embedded in surface observations: mesoscale eddies, thermocline displacement patterns, and wind-driven vertical mixing dynamics.

**In-Situ Neural 4D-Var Prompting (in_situ_prompting.py)**

When live Argo floats are available for the current day, they are encoded as prompt tokens and fused with satellite embeddings via a cross-attention mechanism:

```
H_assimilated = Softmax(Q_satellite * K_argo^T / sqrt(d)) * V_argo + H_satellite
```

This acts as a learned spatial interpolation operator that calibrates the satellite-derived reconstruction against actual subsurface measurements at float positions, then propagates corrections across the basin.

**Physics-Guided Reconstruction (hybrid_reconstructor.py)**

Rather than directly predicting temperature at 15 arbitrary depth levels, the model predicts baroclinic normal mode amplitude coefficients a_m(x, y):

```
T(x, y, z) = T_clim(x, y, z) + sum_{m=1}^{5} a_m(x, y) * Phi_m(z)
```

Where Phi_m(z) are precomputed orthogonal vertical eigenfunctions from the Sturm-Liouville decomposition. This formulation:
- Guarantees gravitational stability (N^2 >= 0) by construction
- Respects known ocean vertical structure (mixed layer, thermocline, deep water)
- Reduces the prediction space from 15 independent depths to 5 physically-meaningful mode coefficients

**Embedding Exporter (latent_embedder.py)**

Dedicated module to extract, save, and visualize intermediate satellite embeddings. This is required because the problem statement specifically calls for a "Satellite Embedding-Based Framework" -- evaluators need to inspect that the embeddings capture meaningful ocean dynamical features.

**Physics Loss Function (physics_loss.py)**

```
L_total = L_MSE + lambda_stab * L_stability
```

Where:
- L_MSE is the depth-weighted mean squared error with inverse-variance weighting (1 / sigma_z^2)
- L_stability penalizes non-physical buoyancy inversions (dT/dz > 0) below the mixed layer

### 3.3. Evaluation Framework (src/evaluation/)

| Module | Purpose |
|--------|---------|
| `metrics.py` | Computes depth-stratified RMSE, MAE, Bias, and Pearson correlation across 15 standard levels |
| `benchmark_report.py` | Generates formatted 15-depth benchmark scorecard for evaluation |

### 3.4. Downstream Applications (src/domain/)

The primary output (3D subsurface temperature field) enables downstream civilian disaster management applications relevant to the INCOIS mandate:

| Module | Purpose | Relevance to INCOIS |
|--------|---------|---------------------|
| `cyclone_tchp.py` | Tropical Cyclone Heat Potential and D26 isotherm depth | Cyclone rapid intensification forecasting -- core INCOIS disaster warning mandate |

### 3.5. Operational API and Web Interface

- `src/api/main.py`: FastAPI backend serving reconstruction endpoints, embedding inspection, depth-slice queries, and GeoJSON exports
- `web/`: Interactive WebGL digital twin dashboard with 2D/3D ocean thermal visualization over Leaflet/Plotly maps

---

## 4. Training and Validation Data Flow

```
Training Phase:
  Input:  OSTIA SST + SMAP SSS + DUACS SSH + OSCAR Currents + CCMP Winds (Surface, 0.25 deg)
  Target: GLORYS12V1 Reanalysis Subsurface Temperature (15 Depths, 0.25 deg)

Validation Phase:
  Predictions vs. Independent INCOIS Gridded ARGO Observations
  Metrics: Depth-wise RMSE, Bias, Pearson Correlation for each of the 15 standard levels

Real-Time Inference:
  Input:  Today's satellite surface observations + live Argo float casts (optional)
  Output: Reconstructed 3D temperature field T(x, y, z, t) with uncertainty bounds
```

---

## 5. Input and Target Datasets (Per INCOIS Problem Statement)

| Variable | Product | Native Resolution | Source |
|----------|---------|-------------------|--------|
| SST | OSTIA | 0.05 deg, daily | https://doi.org/10.48670/moi-00168 |
| SSS | SMAP / SMOS | 0.125 deg, daily | https://doi.org/10.48670/moi-00051 |
| SSH / SLA | DUACS | 0.25 deg, daily | https://doi.org/10.48670/moi-00145 |
| Currents (U, V) | OSCAR L4 | 0.25 deg, daily | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0 |
| Winds (U, V) | CCMP / ASCAT | 0.25 deg, 6-hourly | https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 |
| **Target**: Subsurface Temp | GLORYS12V1 | 0.083 deg, daily | https://doi.org/10.48670/moi-00021 |
| **Validation**: In-Situ | Gridded ARGO (INCOIS LAS) | Point profiles | INCOIS Live Access Server |

All datasets are regridded to 0.25 deg x 0.25 deg daily resolution during preprocessing.

---

## 6. Key Design Decisions

1. **Why Mamba SSM over Vision Transformer?** Linear O(N) scaling is critical for operational 0.25 deg basin-scale inference (101 x 241 grid = 24,341 spatial points). ViT's O(N^2) attention becomes prohibitive at this resolution.

2. **Why Baroclinic Modal Synthesis over Direct Regression?** Predicting 5 mode coefficients instead of 15 independent depths enforces physical consistency and prevents unphysical density inversions that corrupt downstream heat content calculations.

3. **Why Neural 4D-Var over Traditional Data Assimilation?** Traditional 4D-Var requires explicit numerical ocean model adjoint integration. Our learned cross-attention approach achieves comparable assimilation quality at a fraction of the computational cost, making it feasible for real-time operational use.

4. **Why Export Embeddings?** The problem statement specifically requires a "Satellite Embedding-Based" framework. Exporting and visualizing the intermediate latent representations demonstrates that the model learns meaningful ocean dynamical features, not just surface-to-depth regression.
