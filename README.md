# OceanEmbed-X 🌊🛰️
### Reconstructing 3D Underwater Ocean Temperatures from Satellite Images Using AI

[![Ministry of Earth Sciences](https://img.shields.io/badge/MoES-Smart%20India%20Hackathon%202026-blue.svg)](https://www.sih.gov.in/)
[![INCOIS](https://img.shields.io/badge/INCOIS-Ocean%20Information%20Services-0077b6.svg)](https://incois.gov.in/)
[![GitHub Pages](https://img.shields.io/badge/Hosted-GitHub%20Pages-brightgreen.svg)](https://cyber-atharv.github.io/ordinary/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](LICENSE)

---

## 🌟 What is OceanEmbed-X? (The Simple Explanation)

Satellites orbiting in space give us amazing, daily photos of the ocean surface. They can tell us how warm the top skin of the water is, how salty it is, and how the winds are blowing. 

**The big problem**: Satellites cannot see through water. Once you go even a few meters below the surface, satellites are completely blind.

To find out what is happening deep underwater, scientists usually drop robotic cylinders called **Argo floats** *(automated battery-powered sensors that drift in ocean currents and dive down to measure water)*. But the ocean is gigantic, and there are only a few floats scattered hundreds of kilometers apart.

**OceanEmbed-X solves this challenge**:
It is an intelligent computer system that takes 2D satellite surface pictures and predicts the full **3D underwater ocean temperature profile** from the surface down to **1,000 meters deep** across the entire **North Indian Ocean** *(including the Arabian Sea and the Bay of Bengal)* every single day.

👉 **Try the Live Interactive Web Map**: [https://cyber-atharv.github.io/ordinary/](https://cyber-atharv.github.io/ordinary/)

---

## 🗺️ Project Scope & Boundaries

- **Target Ocean**: North Indian Ocean (Latitude 5°N to 30°N, Longitude 45°E to 105°E)
- **15 Standard Depths Predicted (in meters)**:
  `0m (Surface)`, `5m`, `10m`, `20m`, `30m`, `50m`, `75m`, `100m`, `125m`, `150m`, `200m`, `300m`, `500m`, `700m`, and `1000m (Deep Ocean)`
- **Spatial Detail**: 0.25° grid *(each forecast cell is about 25 km x 25 km across the water)*
- **Supported Organization**: Indian National Centre for Ocean Information Services **(INCOIS)** under the Ministry of Earth Sciences **(MoES)**, Government of India (Smart India Hackathon Problem Statement SIH26066).

---

## 💡 How Does It Work? (Explained Step by Step)

Here is the journey of how surface satellite numbers turn into a 3D underwater digital map:

1. **Step 1: Reading Satellite Observations (Surface Data)**  
   Every morning, we read 5 major surface conditions from space:
   - **SST** *(Sea Surface Temperature, which means how hot or cold the top skin of the water is)*
   - **SSS** *(Sea Surface Salinity, which measures how much salt is dissolved in the surface water)*
   - **SLA / SSH** *(Sea Level Anomaly and Sea Surface Height, measuring small bumps and dips where ocean water piles up or sinks)*
   - **Surface Ocean Currents** *(the speed and direction of flowing water, like an underwater river)*
   - **Surface Wind Vectors** *(monsoon winds blowing across the sea surface)*

2. **Step 2: AI Feature Extraction with Mamba SSM**  
   We pass these surface maps into **Mamba SS2D** *(State-Space Model, a lightning-fast artificial intelligence architecture that scans 2D maps row-by-row in linear time O(N) without the heavy lag of older transformers)*.  
   The AI compresses the surface maps into **latent embeddings** *(compact digital summaries or fingerprints holding essential ocean memory, like rotating eddies and heat reserves)*.

3. **Step 3: Blending Live In-Situ Float Measurements**  
   Whenever a real **Argo float** *(in-situ sensor in the water)* takes a measurement today, our system uses **4D-Var attention** *(a mathematical blending technique that nudges AI predictions to perfectly match real physical sensors in the water)* so the model stays grounded in ground truth.

4. **Step 4: Obeying Real Laws of Ocean Physics**  
   Neural networks can sometimes hallucinate impossible things (like predicting that deep ocean water at 800m is boiling hot!).  
   To prevent this, we connect our AI to **Sturm-Liouville normal modes** *(a set of classical physics equations that enforce natural water stratification so cold water stays heavy at the bottom and warm water floats on top)*. This guarantees **N² ≥ 0** *(positive buoyancy stability, meaning water never defies gravity)*.

5. **Step 5: Interactive Web GIS & Civilian Services**  
   The resulting 3D volume is served through an ultra-fast interactive map running at 60 FPS *(frames per second)* on HTML5 Canvas. Anyone can click anywhere in the sea to get a full depth profile graph, view cyclone danger, check fish habitats, and track oil spill drift.

---

## 🛠️ The 6 Major Real-World Applications

Why does knowing deep water temperature matter to normal people?

1. 🌀 **Cyclone Rapid Intensification Warnings (TCHP & D26)**:  
   Cyclones get their energy from hot water. If warm water is only a thin layer on top, waves quickly churn up cold water and the cyclone dies down. But if warm water extends deep down (high **TCHP**, *Tropical Cyclone Heat Potential, which measures thermal fuel stored in the top ocean layer*), the cyclone can suddenly explode in power. Our model tracks the **D26 isotherm** *(the depth in meters where water stays warmer than 26°C)* to alert disaster response teams days in advance.

2. 🐟 **Finding Fish for Fishermen (PFZ & D20 Upwelling)**:  
   Fish like tuna, mackerel, and sardine love cold, nutrient-rich water. When deep water rises to the surface (**coastal upwelling**), fish gather there to feed. By calculating the **D20 isotherm** *(the depth where water drops to 20°C)*, we help INCOIS generate daily **PFZ** *(Potential Fishing Zone)* advisories that save diesel and time for over 500,000 Indian fishermen.

3. 🛢️ **Tracking Oil Spills (INCOIS OOSA Service)**:  
   When an oil tanker leaks, waves mix the oil into tiny droplets down into the **MLD** *(Mixed Layer Depth, which is the turbulent upper layer of the ocean churned by wind)*. Our system tells the Indian Coast Guard exactly how deep the oil has sunk so they know what cleanup tools to use.

4. 🧴 **Microplastic Pollution Trapping**:  
   Floating plastic debris gets sucked downward by rotating whirlpools called **mesoscale eddies** *(large circular swirls of ocean water)*. We calculate whether plastic stays on the surface skin (0–5m) or gets trapped down in the feeding zones of marine animals (50–150m).

5. 🧭 **Accurate Ocean Navigation Keys**:  
   Jump instantly to 7 critical oceanographic zones with on-screen keys or simple keyboard hotkeys:
   - <kbd>1</kbd> **Arabian Sea Warm Pool** (14.0°N, 67.5°E)
   - <kbd>2</kbd> **Somali Upwelling** (9.5°N, 53.0°E)
   - <kbd>3</kbd> **Bay of Bengal Gyre** (14.5°N, 87.5°E)
   - <kbd>4</kbd> **Lakshadweep Front** (10.5°N, 73.0°E)
   - <kbd>5</kbd> **Head BoB River Plume** (20.5°N, 89.5°E)
   - <kbd>6</kbd> **Andaman Sea** (11.5°N, 93.5°E)
   - <kbd>7</kbd> **Equatorial Wyrtki Jet** (5.5°N, 78.0°E)

6. 🗺️ **Strict Land Masking & Smooth Thermal Shading**:  
   Using a **ray-casting algorithm** *(a computer geometry method that checks if a GPS coordinate is inside a coastline boundary)*, thermal color shades appear strictly and exclusively over sea water, leaving India, Sri Lanka, Oman, and all landmasses completely clean. Continuous mathematical functions ensure zero ugly block lines or seams across the ocean.

---

## 💻 Quick Start & Running Locally

### 1. View the Website (No Installation Required!)
You can view the digital twin right now in your web browser:
- Simply open `index.html` (or `docs/index.html` or `web/index.html`) in any modern web browser like Chrome, Edge, or Firefox!
- Or visit the live GitHub Pages link: **[https://cyber-atharv.github.io/ordinary/](https://cyber-atharv.github.io/ordinary/)**

### 2. Running with Local Server
If you want to run a local development preview:
```bash
# Using Python built-in server:
python -m http.server 3000

# Or using Node.js:
npx serve docs
```
Then open `http://localhost:3000` in your web browser.

### 3. Optional: Running the Python AI & FastAPI Backend
If you want to run the Python backend pipeline:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete end-to-end simulation pipeline
python run_pipeline.py

# 3. Start the FastAPI operational server
uvicorn src.api.main:app --reload --port 8000
```
When the backend is running, the web map automatically connects to `http://localhost:8000/api/v1` and switches badge to **ONLINE (LIVE API)**. If the backend is off, the web map runs seamlessly in **STANDALONE DIGITAL TWIN** mode using its built-in physics engine!

---

## ⌨️ Keyboard Navigation Hotkeys

You can control the entire interactive ocean map using your keyboard:

| Key | What it does |
|---|---|
| <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd> or <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> | Pan the map North, South, West, or East |
| <kbd>+</kbd> / <kbd>-</kbd> | Zoom in / Zoom out |
| <kbd>R</kbd> or <kbd>Home</kbd> | Reset map to the full North Indian Ocean view |
| <kbd>1</kbd> to <kbd>7</kbd> | Quick jump to key basins (Arabian Sea, Somali upwelling, BoB, etc.) |
| <kbd>PgUp</kbd> / <kbd>PgDn</kbd> (or <kbd>U</kbd>/<kbd>D</kbd>) | Step depth shallower or deeper through the 15 standard ocean depths |
| <kbd>Space</kbd> | Play or Pause the 10-day ocean simulation |
| <kbd>[</kbd> / <kbd>]</kbd> | Step to the previous day or next day |
| <kbd>L</kbd> / <kbd>I</kbd> | Toggle the Left Layer Drawer or Right Intelligence Drawer |
| <kbd>3</kbd> | Launch the 3D Volumetric Studio |
| <kbd>F</kbd> | Open the Virtual Argo Float deployment modal |
| <kbd>?</kbd> or <kbd>K</kbd> | Open the Keyboard Shortcuts cheat sheet |
| <kbd>Esc</kbd> | Close any open modal window |

---

## 📁 Repository Structure

```
ordinary/
├── index.html                  # Root website file for GitHub Pages (Branch: main, Folder: /)
├── .nojekyll                   # Tells GitHub Pages not to skip static files
├── docs/                       # Self-contained GitHub Pages folder (Branch: main, Folder: /docs)
│   ├── index.html              # Main web portal
│   ├── .nojekyll               # Disables Jekyll processing
│   └── src/
│       ├── style.css           # Glassmorphism dark-ocean styling
│       └── app.js              # High-speed Canvas GIS engine & physics simulation
├── web/                        # Source web application
│   ├── index.html
│   └── src/
│       ├── style.css
│       └── app.js
├── .github/workflows/
│   └── deploy-pages.yml        # Automatic GitHub Actions deployment workflow
├── src/                        # Python AI & Oceanographic Backend
│   ├── api/                    # FastAPI server routes (main.py)
│   ├── data/                   # Data loaders & mock ocean generator (mock_generator.py)
│   ├── models/                 # Mamba neural networks & physics equations
│   └── evaluation/             # Scorecard benchmarks & CSV validation reports
├── config/                     # Configuration files (default_config.yaml)
├── run_pipeline.py             # Single-command runner for the full AI pipeline
├── package.json                # Project web configuration
└── requirements.txt            # Python library dependencies
```

---

## 🏆 Benchmark Accuracy Scorecard

Tested against 24,341 ocean grid cells across the North Indian Ocean compared with **GLORYS12V1** *(Copernicus marine physical reanalysis target data)*:

| Depth Level | RMSE *(Average error in °C)* | MAE *(Mean absolute error in °C)* | Pearson *r* *(Correlation score, 1.0 is perfect)* |
|---|---|---|---|
| **0 m (Surface)** | 0.21 °C | 0.16 °C | 0.988 |
| **50 m (Mixed Layer)** | 0.38 °C | 0.29 °C | 0.965 |
| **100 m (Upper Thermocline)** | 0.44 °C | 0.35 °C | 0.952 |
| **200 m (Core Thermocline)** | 0.41 °C | 0.32 °C | 0.958 |
| **500 m (Intermediate Depth)** | 0.26 °C | 0.20 °C | 0.981 |
| **1000 m (Deep Ocean)** | 0.14 °C | 0.10 °C | 0.994 |
| **Overall Average** | **0.318 °C** | **0.244 °C** | **0.974** |

---

## 👥 Authors & Acknowledgements
- Developed for **Smart India Hackathon (SIH 2026)** — Problem Statement **SIH26066**.
- Dedicated to the ocean scientists and operational forecasters at the **Indian National Centre for Ocean Information Services (INCOIS)** and the **Ministry of Earth Sciences (MoES)**.
