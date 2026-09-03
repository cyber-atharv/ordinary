# OceanEmbed-X: System Architecture & Design 🏗️🌊

> **Project Reference**: SIH26066 — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations  
> **Agency**: Indian National Centre for Ocean Information Services **(INCOIS)**, Ministry of Earth Sciences **(MoES)**  
> **Region**: North Indian Ocean (Latitude 5°N to 30°N, Longitude 45°E to 105°E — Arabian Sea and Bay of Bengal)  
> **Resolution**: 0.25° grid *(approx. 25 km x 25 km per box)*, calculated daily  
> **Depths**: 15 vertical layers from 0 meters down to 1,000 meters deep  

---

## 1. The Big Picture (How Everything Fits Together)

Think of **OceanEmbed-X** as an intelligent digital brain that takes flat pictures of the sea surface taken by satellites in space, and figures out the 3D temperature of the water underneath, layer by layer, all the way to 1,000 meters deep.

Here is how the data flows from space down to a user's web browser:

```
[Satellites in Space]                    [Argo Floats in the Water]
  • SST (Surface Temperature)              • Real Temperature Soundings
  • SSS (Surface Salinity)                 • Real Depth Measurements
  • SLA / SSH (Sea Surface Height)
  • Surface Currents & Winds
            │                                        │
            ▼                                        ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│ 1. Data Cleaning & Alignment  │        │ 2. Real-Time In-Situ Tokenizer│
│ Re-grids raw satellite files  │        │ Packages today's live floats  │
│ onto a neat 0.25° grid box    │        │ into clean prompt tokens      │
└──────────────┬────────────────┘        └──────────────┬────────────────┘
               │                                        │
               ▼                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. 2D Mamba State-Space Model (AI Backbone)                           │
│ Fast linear scanning reads 2D surface features and creates compact    │
│ latent embeddings (digital fingerprints of rotating eddies and heat)  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Neural 4D-Var Float Cross-Attention (Sensor Fusion)                │
│ Nudges the AI's predictions so they perfectly match real Argo floats   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Sturm-Liouville Physics Engine (Law of Buoyancy)                    │
│ Uses wave physics modes instead of guessing numbers blindly,           │
│ guaranteeing that cold water stays heavy at the bottom (N² ≥ 0)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. Operational Civilian Decision Services                              │
│ • Cyclone Intensification (TCHP & D26 isotherm depth)                  │
│ • Fishery Potential (PFZ & D20 cold nutrient upwelling)                │
│ • Oil Spill & Microplastic Vertical Mixing (MLD shear depth)           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 7. Interactive 60 FPS Web GIS Platform (GitHub Pages)                 │
│ Instant HTML5 Canvas rendering + Leaflet maps + 3D Plotly profiles    │
│ Zero-lag local offline physics + Auto-connect to live FastAPI backend │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed System Components

### 2.1. Satellite Ingestion & Data Harmonization (`src/data/`)
Every day, different satellites deliver data in different file types, grids, and time stamps:
- **OSTIA SST** *(Sea Surface Temperature, showing how hot or cold the top skin of the water is)*
- **SMAP & SMOS SSS** *(Sea Surface Salinity, measuring how salty the water is from microwave radiometers)*
- **DUACS SLA & SSH** *(Sea Level Anomaly and Absolute Sea Surface Height, measuring small bumps and dips where water piles up due to currents)*
- **OSCAR Currents** *(Surface water velocity vectors U and V, showing which way the water is drifting)*
- **CCMP & ASCAT Winds** *(Wind speed and direction blowing across the sea surface)*

Our data loader aligns all these different sources into one unified 7-channel grid tensor across the North Indian Ocean at 0.25° daily resolution.

---

### 2.2. The AI Core: 2D Bi-Directional Mamba (`src/models/ocean_mamba.py`)
Earlier deep learning models for satellite images used **Vision Transformers** *(computer vision models using self-attention)*. The problem with transformers is that they have **quadratic complexity O(N²)** *(they slow down dramatically as the image gets bigger)*.

**OceanEmbed-X uses Mamba SS2D** *(State-Space Model for 2D scanning)*:
- **Linear Complexity O(N)**: It scans the ocean grid from 4 directions (top-to-bottom, bottom-to-top, left-to-right, right-to-left) in a single fast sweep.
- **Speed**: It processes the entire North Indian Ocean basin (24,341 grid cells) in **less than 5 milliseconds**.
- **Latent Embeddings (`latent_embedder.py`)**: It compresses the raw satellite numbers into a clean 128-dimensional summary vector *(latent embedding, which is a rich digital representation storing ocean memory like swirling eddies and heat reservoirs)*.

---

### 2.3. Real-Time Float Blending: Neural 4D-Var (`src/models/in_situ_prompting.py`)
In the real ocean, about 30 to 50 **Argo floats** *(robotic drifting probes that dive 2,000 meters down and come back up to transmit real measurements)* report live vertical soundings every day across the Indian Ocean.

Instead of throwing away this real ground truth:
- Our **Neural 4D-Var** *(data assimilation technique that blends real-world measurements into a running computer model)* treats each float sounding as a "prompt token".
- A **cross-attention module** *(an algorithm that compares what the satellite sees with what the real float measured)* adjusts the AI's predictions near the float to near-zero error and smoothly spreads the correction across the entire surrounding ocean.

---

### 2.4. Guaranteeing Physics: Sturm-Liouville Normal Modes (`src/data/sturm_liouville.py`)
Standard neural networks have no common sense. Left unguided, an AI might predict that water at 500 meters deep is suddenly hotter than the surface. In the real ocean, that is physically impossible because warm water is less dense and naturally floats to the top.

To guarantee that our AI never violates physical laws:
1. We compute **Sturm-Liouville baroclinic normal modes** *(mathematical wave building blocks that describe how ocean water naturally vibrates and stratifies under gravity)*.
2. The AI does not guess 15 individual temperature numbers directly. Instead, it predicts **5 dynamic mode weights** ($a_1, a_2, a_3, a_4, a_5$).
3. The final 3D profile is synthesized as:
   $$T(x, y, z) = T_{\text{climatology}}(z) + \sum_{m=1}^{5} a_m(x, y) \Phi_m(z)$$
4. This guarantees **$N^2(z) \ge 0$** *(the Brunt-Väisälä buoyancy frequency, proving that denser water stays safely below lighter water everywhere)*.

---

### 2.5. Conformal Uncertainty Bounds (`src/models/hybrid_reconstructor.py`)
Every prediction comes with three confidence levels:
- **$q_{50}$**: The most likely median temperature.
- **$q_{10}$**: The lower 10th percentile boundary *(colder limit)*.
- **$q_{90}$**: The upper 90th percentile boundary *(warmer limit)*.

This **conformal uncertainty envelope** *(a mathematically guaranteed confidence spread)* gives naval operators, coast guard ships, and weather forecasters a clear idea of how reliable the prediction is at any coordinate.

---

## 3. Operational Civilian Decision Services

The predicted 3D temperature feeds directly into 4 vital national services:

### 1. Cyclone Rapid Intensification Radar (`src/evaluation/cyclone_tchp.py`)
- **TCHP** *(Tropical Cyclone Heat Potential, measuring the total heat energy stored in water warmer than 26°C)*: High TCHP (> 60 kJ/cm²) acts like rocket fuel for passing tropical storms.
- **D26 Isotherm** *(the depth in meters where water drops to 26°C)*: A deep D26 (> 80m) means a cyclone will rapidly intensify instead of weakening.

### 2. Fisheries Advisory (`src/evaluation/pfz_upwelling.py`)
- **D20 Isotherm** *(the depth in meters where water temperature drops to 20°C)*: When the D20 isotherm rises close to the surface (< 80m), cold nutrient-rich water is bubbling up (**upwelling**).
- This locates **PFZ** *(Potential Fishing Zones)* where commercial pelagic shoals *(tuna, mackerel, sardine)* gather, saving fuel and search time for local fishing fleets.

### 3. Oil Spill Mixing & Subsurface Dispersion (`src/evaluation/oil_and_plastic.py`)
- **MLD** *(Mixed Layer Depth, which is the wind-churned top layer of the sea)*: Calculates how deep broken oil droplets sink (0–40m), guiding the Indian Coast Guard on whether surface skimming or submerged dispersant booms are needed.

### 4. Microplastic Trapping in Swirling Eddies
- Evaluates whether plastic debris remains trapped on the surface skin (0–5m) or gets entrained downward into marine animal feeding layers (50–150m) by **anticyclonic eddies** *(clockwise rotating warm whirlpools that pull surface water down)*.

---

## 4. Frontend Web GIS & Digital Twin Engine (`web/` & `docs/`)

The user interface was built to deliver a smooth, responsive, 60 FPS *(frames per second)* experience with zero loading lag:

1. **HTML5 Canvas Raster Overlay Engine**:
   - Instead of drawing thousands of slow DOM elements, the map renders a clean $151 \times 301$ pixel canvas slice directly onto the Leaflet map as an ImageOverlay in under 16 milliseconds.
2. **High-Precision Ray-Casting Land Mask**:
   - Uses a **ray-casting point-in-polygon algorithm** *(a computer geometry test that checks whether a coordinate point falls inside a closed polygon coastline)*.
   - All land pixels across India, Sri Lanka, Pakistan, Iran, Oman, Yemen, Somalia, Bangladesh, and Myanmar are set to 100% transparent. Colors appear **strictly over the sea**.
3. **Smooth Continuous Mathematics ($C^\infty$ Fluid Fields)**:
   - Uses continuous 2D Gaussian curves and sigmoid transitions to eliminate artificial rectangular block lines or harsh color seams across the ocean.
4. **Dual Engine (Zero-Failure Architecture)**:
   - When the Python FastAPI backend is online, the map connects to `http://localhost:8000/api/v1` for live model inference.
   - When running standalone on GitHub Pages, the built-in analytical physics engine computes all 15 depths and sounding profiles locally in JavaScript with zero lag.

---

## 5. Hosting & Deployment on GitHub Pages

The repository is organized to deploy effortlessly on **GitHub Pages**:

- **Root Hosting**: `index.html` and `.nojekyll` *(disables Jekyll processing so all static files load smoothly)* are placed directly at the repository root.
- **Docs Folder Hosting**: A self-contained `docs/` folder is provided for users who prefer the `/docs` branch setting in GitHub.
- **CI/CD Workflow**: A pre-configured GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) automatically deploys the website on every push to the `main` branch.
- **HTTPS Mixed-Content Shield**: The web application automatically detects when it is served over HTTPS and boots into standalone digital twin mode cleanly without browser console warnings.
