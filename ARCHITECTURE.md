# OceanEmbed-X: System Architecture & Technical Specifications (SIH26066)

> **Problem Statement**: SIH26066 -- Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations  
> **Organization**: Indian National Centre for Ocean Information Services (INCOIS), Ministry of Earth Sciences (MoES)  
> **Category**: Software | **Theme**: Disaster Management & Marine Ecosystem Services  
> **Target Region**: North Indian Ocean ($5^\circ\text{N}\text{ to }30^\circ\text{N}, 45^\circ\text{E}\text{ to }105^\circ\text{E}$ -- Arabian Sea & Bay of Bengal)  
> **Grid**: $0.25^\circ \times 0.25^\circ$ Daily Temporal Resolution (101 x 241 = 24,341 spatial cells)  
> **Standard Depths (m)**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`  

---

## 1. Problem Context & Scientific Mandate

Subsurface ocean temperature ($0\text{--}1000\text{m}$) is a fundamental geophysical state variable governing upper-ocean heat content, planetary circulation, air-sea interaction, and marine ecosystems. In-situ observing systems (Argo profiling floats, moored buoys, gliders) provide sparse point vertical soundings. Satellite sensors provide continuous basin-scale coverage but are limited strictly to the 2D surface skin.

The primary objective of **OceanEmbed-X** is to learn the nonlinear physical mapping from multi-modal 2D satellite surface observations to the continuous 3D subsurface thermal field across the North Indian Ocean at $0.25^\circ$ daily resolution using a **Satellite Embedding-Based Deep Learning Framework**.

---

## 2. End-to-End Master Pipeline Architecture

```
Surface Satellite Observations (7 Channels, 0.25 deg Daily Grid)
  SST (OSTIA) | SSS (SMAP/SMOS) | SSH/SLA (DUACS) | Currents U,V (OSCAR) | Winds U,V (CCMP/ASCAT)
        |
        v
 +-------------------------------------------------------------------------+
 | Step 1: Preprocessing & Multi-Source Harmonization (src/data/)          |
 |  - Spatial regridding to 0.25 deg daily via bilinear xarray/CDO         |
 |  - Land-sea coordinate masking matching INCOIS/GEBCO coastal bounds     |
 +-------------------------------------------------------------------------+
        |
        v
 +-------------------------------------------------------------------------+
 | Step 2: 2D Mamba State-Space Embedding Engine (src/models/ocean_mamba.py)|
 |  - Bi-directional 2D Selective Scan (SS2D) with linear O(N) complexity   |
 |  - Generates compact latent representation z in R^128 x H x W            |
 |  - Exportable latent features (src/models/latent_embedder.py)           |
 +-------------------------------------------------------------------------+
        |                                       |
        v                                       v
 +-------------------------------+  +---------------------------------------+
 | Step 3a: Analytical           |  | Step 3b: In-Situ Neural 4D-Var        |
 | Sturm-Liouville Normal Modes  |  | Float Assimilation                    |
 | (src/data/sturm_liouville.py) |  | (src/models/in_situ_prompting.py)     |
 |                               |  |                                       |
 | Solves vertical eigenfunctions|  | Cross-attention fuses today's live    |
 | Phi_m(z) under Rossby radii   |  | Argo float tokens into latent space   |
 | constraints (N^2 >= 0)        |  | to calibrate satellite embeddings     |
 +-------------------------------+  +---------------------------------------+
        |                                       |
        +-------------------+-------------------+
                            |
                            v
 +-------------------------------------------------------------------------+
 | Step 4: Physics-Guided 3D Reconstruction (src/models/hybrid_reconstructor)|
 |  - Predicts 5 dynamic baroclinic mode weights a_m(x, y)                  |
 |  - Synthesizes T(x,y,z) = T_clim(z) + sum_{m=1}^5 a_m(x,y) * Phi_m(z)  |
 |  - Conformal multi-quantile uncertainty bounds (10th, 50th, 90th spread)|
 |  - Output: T(x, y, z, t) at 15 standard depth levels (0-1000m)          |
 +-------------------------------------------------------------------------+
        |
        +──────────────────────────┬──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
 [Disaster Services]       [Ecosystem Services]      [Observation Optimization]
 • Cyclone TCHP & D26      • PFZ & D20 Upwelling     • Active Sampling Optimizer
   (cyclone_tchp.py)         (pfz_upwelling.py)        (active_sampling.py)
 • INCOIS OOSA Oil Spill   • Microplastic Submergence
   (oil_and_plastic.py)      (oil_and_plastic.py)
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                                   v
 +-------------------------------------------------------------------------+
 | Step 5: Verification & Benchmarking (src/evaluation/)                   |
 |  - 15-Depth Scorecard: RMSE, MAE, Bias, Pearson r per layer             |
 |  - Automated CSV Exporter (src/evaluation/benchmark_report.py)          |
 +-------------------------------------------------------------------------+
        |
        v
 +-------------------------------------------------------------------------+
 | Step 6: Operational Serving & Interactive Web GIS (src/api/ + web/)     |
 |  - FastAPI High-Speed REST Server (<10ms inference latency)             |
 |  - 5-Tab Interactive Web GIS & 3D Volumetric Digital Twin Dashboard     |
 |  - Single-Command Operational Runner (run_pipeline.py)                  |
 +-------------------------------------------------------------------------+
```

---

## 3. Module Details & Mathematical Formulations

### 3.1. Data Ingestion and Harmonization (`src/data/`)

| Module | Purpose | Source / Reference |
|---|---|---|
| `glorys_loader.py` | Downloads and regrids GLORYS12V1 reanalysis targets (15 depths) | [Copernicus Marine](https://doi.org/10.48670/moi-00021) |
| `satellite_fetcher.py` | Fetches OSTIA SST, SMAP SSS, DUACS SSH, OSCAR currents, CCMP winds | PODAAC / Copernicus |
| `incois_argo_pipeline.py` | Retrieves gridded Argo fields from INCOIS Live Access Server (LAS) | INCOIS LAS |
| `argo_pipeline.py` | Retrieves real-time point Argo float casts for live assimilation | `argopy` GDAC API |
| `sturm_liouville.py` | Analytical Sturm-Liouville vertical eigenvalue solver | Rossby Normal Modes |
| `mock_generator.py` | High-fidelity offline physical test array generator | Zero-download dev mode |

### 3.2. Deep Learning Backbone (`src/models/`)

#### 1. 2D Selective State-Space Embedding (`ocean_mamba.py`)
Rather than quadratic Vision Transformers $\mathcal{O}(N^2)$, the backbone uses a continuous 2D Selective State-Space Model (SS2D) with linear computational complexity $\mathcal{O}(N)$. Given input surface tensor $\mathbf{X} \in \mathbb{R}^{B \times 7 \times H \times W}$, it executes 4-directional spatial scans:
$$h_k = \mathbf{\bar{A}} h_{k-1} + \mathbf{\bar{B}} x_k, \quad y_k = \mathbf{C} h_k + \mathbf{D} x_k$$
Producing a compact 128-dimensional latent embedding $z \in \mathbb{R}^{B \times 128 \times H \times W}$ that encodes mesoscale eddy vorticity and thermocline displacement.

#### 2. In-Situ Neural 4D-Var Cross-Attention (`in_situ_prompting.py`)
Fuses $M$ sparse live Argo float soundings $\{\mathbf{p}_i = (lat_i, lon_i, T_i(z))\}_{i=1}^M$ with satellite embeddings:
$$\mathbf{H}_{\text{assimilated}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{satellite}} \mathbf{K}_{\text{argo}}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\text{argo}} + \mathbf{H}_{\text{satellite}}$$
Guarantees near-zero error at active float positions while propagating corrections across the basin.

#### 3. Analytical Baroclinic Modal Synthesis (`hybrid_reconstructor.py` & `sturm_liouville.py`)
Solves the vertical Sturm-Liouville eigenvalue problem for stratified fluid dynamics:
$$\frac{d}{dz} \left( \frac{1}{N^2(z)} \frac{d\Phi_m}{dz} \right) + \frac{1}{c_m^2} \Phi_m(z) = 0$$
The 3D continuous temperature profile is synthesized as:
$$T(x, y, z) = T_{\text{clim}}(z) + \sum_{m=1}^{5} a_m(x, y) \Phi_m(z)$$
Enforcing gravitational stability ($N^2(z) \ge 0$) by construction and eliminating unphysical density inversions.

#### 4. Satellite Latent Embedding Exporter (`latent_embedder.py`)
Exports intermediate latent representations $z \in \mathbb{R}^{128}$ to compressed `.npz` archives and computes energy norms across 8 semantic feature groups (Vorticity, Rossby phase, Divergence, Heat flux memory).

#### 5. Physics-Guided Loss Function (`physics_loss.py`)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE-norm}} + \lambda_{\text{stab}} \mathcal{L}_{\text{stability}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smoothness}}$$
Where $\mathcal{L}_{\text{MSE-norm}} = \sum_{k=1}^{15} \frac{1}{\sigma_k^2} (T_{\text{pred}}(z_k) - T_{\text{true}}(z_k))^2$ uses inverse-variance depth weighting.

---

## 4. INCOIS Operational & Ecosystem Domain Modules (`src/domain/`)

The reconstructed 3D temperature field directly powers 4 flagship INCOIS civilian services:

### 4.1. Tropical Cyclone Heat Potential (`cyclone_tchp.py`)
Computes upper-ocean thermodynamic fuel feeding cyclone rapid intensification in the Arabian Sea & Bay of Bengal:
$$\text{TCHP} = \rho c_p \int_{0}^{D_{26}} (T(z) - 26) \, dz \quad [\text{kJ/cm}^2]$$
Where $D_{26}$ is the depth of the $26^\circ\text{C}$ isotherm. $\text{TCHP} > 60\text{ kJ/cm}^2$ triggers Rapid Intensification (RI) early warnings.

### 4.2. Potential Fishing Zone (PFZ) & $D_{20}$ Upwelling (`pfz_upwelling.py`)
Identifies the $20^\circ\text{C}$ isotherm depth ($D_{20}$) and thermocline vertical gradient $\partial T/\partial z$ to detect nutrient-rich upwelling. Outputs:
- $D_{20}$ Thermocline Proxy Depth (m)
- Upwelling Status (Active Divergence / Moderate / Downwelling Warm Pool)
- Recommended Pelagic Fishing Gear Depth (m) for Tuna, Mackerel, and Sardine shoals

### 4.3. INCOIS OOSA: Oil Spill & Microplastic Dispersion (`oil_and_plastic.py`)
Implements physical boundary layer models aligned with the INCOIS **Online Oil Spill Advisory (OOSA)**:
- **Oil Droplet Entrainment Depth**: Scales vertical mixing depth via Mixed Layer Depth ($\text{MLD}$) and wind speed: $z_{\text{droplet}} \sim \text{MLD} \cdot \min(1.0, (U_{\text{wind}}/8.0)^{1.5})$.
- **Marine Microplastic Submergence Zone**: Evaluates whether microplastics remain trapped in the surface neuston layer ($0\text{--}5\text{m}$) or are entrained into the subsurface euphotic zone ($50\text{--}150\text{m}$) by buoyancy frequency $N^2(z)$.

### 4.4. Intelligent Active Sampling Float Optimizer (`active_sampling.py`)
Leverages model multi-quantile uncertainty spread $\Delta T(x, y, z) = T_{90} - T_{10}$ to solve:
$$(x^*, y^*) = \arg\max_{(x, y)} \left[ \sum_{z} w_z \cdot \Delta T(x, y, z) \cdot \prod_{i=1}^M \left( 1 - e^{-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2 R_{\text{excl}}^2}} \right) \right]$$
Recommending optimal GPS drop coordinates for future Argo float and glider deployment by MoES research vessels (*ORV Sagar Nidhi*).

---

## 5. 15-Depth Evaluation Framework (`src/evaluation/`)

Evaluates performance across all 15 discrete depth levels per official INCOIS guidelines:

| Metric | Formula | Description |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{N}\sum (T_{\text{pred}} - T_{\text{true}})^2}$ | Absolute reconstruction error in $^\circ\text{C}$ per depth layer |
| **MAE** | $\frac{1}{N}\sum |T_{\text{pred}} - T_{\text{true}}|$ | Mean absolute deviation in $^\circ\text{C}$ |
| **Bias** | $\frac{1}{N}\sum (T_{\text{pred}} - T_{\text{true}})$ | Systematic warm/cold offset in $^\circ\text{C}$ |
| **Pearson Correlation ($r$)** | $\frac{\text{Cov}(T_{\text{pred}}, T_{\text{true}})}{\sigma_{\text{pred}} \sigma_{\text{true}}}$ | Spatial pattern correlation coefficient |

The scorecard is automatically computed and exported to CSV by `src/evaluation/benchmark_report.py`.

---

## 6. REST API Endpoints & Operational Architecture

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | Grid bounds, depth levels, and system health |
| `/api/v1/predict/profile` | GET | 15-depth reconstruction, 10-90% bounds, MLD, TCHP, PFZ, OOSA |
| `/api/v1/predict/slice` | GET | 2D synoptic map grid for SST, SSS, SLA, Winds, TCHP, D20, or $T_z$ |
| `/api/v1/volume/3d` | GET | 3D volumetric mesh for WebGL isosurface rendering |
| `/api/v1/embeddings/inspect` | GET | 128-channel Mamba latent embedding stats & eddy energy index |
| `/api/v1/domain/pfz` | GET | INCOIS Potential Fishing Zone & $D_{20}$ upwelling advisory |
| `/api/v1/domain/pollution` | GET | INCOIS OOSA oil droplet mixing depth & plastic submergence risk |
| `/api/v1/domain/active_sampling` | GET | Optimal research vessel float deployment recommendations |
| `/api/v1/cyclones/tracks` | GET | Cyclone database with real-time TCHP & RI risk tracking |
| `/api/v1/floats/inject` | POST | In-situ Neural 4D-Var virtual Argo float injection & assimilation |
| `/api/v1/evaluation/benchmark` | GET | Live 15-depth verification benchmark scorecard |
| `/api/v1/export/geojson` | GET | CF-compliant OGC/INCOIS GeoJSON thermal layer exporter |

---

## 7. Official Dataset Specifications

| Variable | Recommended Product | Native Resolution | INCOIS / DOI Citation |
|---|---|---|---|
| **SST** | OSTIA | $0.05^\circ$, daily | [doi:10.48670/moi-00168](https://doi.org/10.48670/moi-00168) |
| **SSS** | SMAP / SMOS | $0.125^\circ$, daily | [doi:10.48670/moi-00051](https://doi.org/10.48670/moi-00051) |
| **SSH / SLA** | DUACS | $0.25^\circ$, daily | [doi:10.48670/moi-00145](https://doi.org/10.48670/moi-00145) |
| **Currents (U, V)** | OSCAR L4 v2.0 | $0.25^\circ$, daily | [PODAAC OSCAR](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0) |
| **Winds (U, V)** | CCMP v3.1 / ASCAT | $0.25^\circ$, 6-hourly | [PODAAC CCMP](https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1) |
| **Target: Subsurface Temp** | GLORYS12V1 | $0.083^\circ$, daily | [doi:10.48670/moi-00021](https://doi.org/10.48670/moi-00021) |
| **Validation: In-Situ** | Gridded ARGO | Point / Gridded | [INCOIS Live Access Server (LAS)](https://incois.gov.in) |

All datasets are regridded to a standardized $0.25^\circ \times 0.25^\circ$ daily grid over the North Indian Ocean ($5^\circ\text{N}\text{ to }30^\circ\text{N}, 45^\circ\text{E}\text{ to }105^\circ\text{E}$).
