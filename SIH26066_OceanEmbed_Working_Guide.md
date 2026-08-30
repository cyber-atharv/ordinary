# SIH26066 — OceanEmbed: Working Guide
### Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations
*Ministry of Earth Sciences (MoES) · Software Track*

---

## 1. Understand the Problem Like a Domain Expert First

Before writing any code, be able to explain this in one paragraph to a judge:

> Satellites can see the ocean surface every day, everywhere — temperature, height, salinity, wind. They cannot see below the surface. But surface conditions are physically linked to what's happening underneath (a warm eddy at the surface usually has a matching thermal signature 200m down). Argo floats measure the actual subsurface temperature, but only at ~4,000 scattered locations worldwide, updated every 10 days. Your job: build a model that learns the surface-to-subsurface relationship from places where both are known (Argo locations), then apply it everywhere satellites have coverage — turning sparse point measurements into a continuous 3D map.

**Why it's hard:** the relationship between surface and subsurface isn't a fixed formula — it changes with season, latitude, ocean currents (e.g., near the Kuroshio or Agulhas current, surface-subsurface coupling behaves very differently than in calm mid-ocean gyres). This is exactly why it's a machine-learning problem and not a lookup table.

**Why judges will like a good solution:** subsurface ocean temperature is used for cyclone intensity prediction, tsunami/monsoon modeling, submarine detection, and fisheries — so a working demo has an obvious "so what."

---

## 2. Data You'll Actually Need

| Purpose | Variable | Source | Access method |
|---|---|---|---|
| Input (surface) | Sea Surface Temperature (SST) | Copernicus Marine / OSTIA | `copernicusmarine` Python package |
| Input (surface) | Sea Surface Height (SSH) / altimetry | Copernicus Marine | `copernicusmarine` |
| Input (surface) | Sea Surface Salinity (SSS) | Copernicus Marine | `copernicusmarine` |
| Input (surface) | Sea Surface Wind (SSW) | Copernicus Marine / ERA5 | `copernicusmarine` or `cdsapi` |
| Ground truth (subsurface) | Temperature profiles at depth | **Argo floats** | `argopy` Python package |
| Validation benchmark | Reanalysis 3D temperature field | GLORYS12V1 / ARMOR3D | `copernicusmarine` |

**Getting started commands:**
```bash
pip install copernicusmarine argopy xarray netCDF4
```

```python
# Surface data (Copernicus Marine)
import copernicusmarine
copernicusmarine.subset(
    dataset_id="cmems_obs-sst_glo_phy_l4_nrt",   # example OSTIA-type SST product
    variables=["analysed_sst"],
    minimum_longitude=60, maximum_longitude=90,   # Indian Ocean box — pick a region, not global
    minimum_latitude=-10, maximum_latitude=25,
    start_datetime="2023-01-01", end_datetime="2023-12-31",
)

# Ground truth (Argo)
from argopy import DataFetcher as ArgoDataFetcher
ds = ArgoDataFetcher().region([60, 90, -10, 25, 0, 2000, '2023-01-01', '2023-12-31']).to_xarray()
```

**Practical tip:** Copernicus Marine requires free registration (an account) before the toolbox will authenticate — do this on day one, not the night before your demo, since registration/verification can take a bit of time.

**Scope it down:** Don't attempt "global ocean." Pick the **Arabian Sea / Indian Ocean** — it's regionally relevant for India, has decent Argo float density, and cuts your data volume dramatically.

---

## 3. Data Preprocessing — the Part Everyone Underestimates

This will likely eat 40% of your total build time. Plan for it explicitly.

1. **Format**: Both Copernicus and Argo data come as NetCDF (`.nc`). Use `xarray` to load — it handles the multi-dimensional (lat, lon, time, depth) structure natively.
2. **Regridding**: Argo points are irregular; satellite data is on a regular grid. You'll need to either (a) interpolate satellite values onto Argo float locations for training, or (b) interpolate Argo profiles onto a regular grid for a gridded target. Approach (a) is simpler for a first model.
3. **Depth binning**: Pick a fixed set of target depths (e.g., 0, 50, 100, 200, 300, 500, 700, 1000 m). Argo profiles have depth samples that don't align across floats — interpolate each profile onto your fixed depth grid first.
4. **Time matching**: Match each Argo profile to the nearest-in-time satellite surface snapshot (same day or within a short window).
5. **Quality control**: Argo has documented QC flags — **use only QC-flagged "good" data** (flag = 1). Skipping this is the single most common student mistake and it quietly wrecks accuracy.
6. **Normalization**: Standardize all inputs (subtract mean, divide by std) before feeding into any neural network — ocean variables have very different scales (SST in °C, SSH in cm, SSS in psu).
7. **Train/val/test split — do this by TIME, not randomly**: e.g., train on 2018–2022, validate on early 2023, test on late 2023. Random splitting leaks information (nearby days are correlated) and will make your accuracy look artificially good.

---

## 4. Model Approaches — Pick Based on Your Timeline

### Tier 1 (Day 1–2, always build this first — your safety net)
**Gradient-boosted trees (XGBoost/LightGBM) per depth level.**
Input: [SST, SSH, SSS, SSW, latitude, longitude, day-of-year] → Output: temperature at one fixed depth. Train a separate model per depth, or one model with depth as an input feature.
- Fast to train, very hard to mess up, gives you a real number to report on day one.
- Published work confirms SSH (from altimetry) is typically the single most predictive surface variable — check that your feature importances agree; if they don't, something's wrong in preprocessing.

### Tier 2 (Day 3–4 — your main deliverable)
**Feedforward neural network (MLP) predicting the full depth profile at once.**
Input: surface variables + lat/lon/time → Output: vector of temperatures at all your chosen depths simultaneously. This captures vertical structure (e.g., thermocline shape) better than predicting depths independently.
- Framework: PyTorch or TensorFlow/Keras.
- 3–4 hidden layers, batch normalization, dropout (0.2–0.3) to prevent overfitting on a modest dataset.

### Tier 3 (Day 5+, if time allows — your differentiator)
**Spatiotemporal attention/graph model** treating nearby grid cells jointly rather than point-by-point, so the model can use neighboring surface patterns (eddies, fronts) as context. This is the approach recent research has shown gives the biggest accuracy jump over simple per-point regression — but it's also the most implementation-heavy, so only attempt it once Tier 1 and 2 are solid and demoable.

**Alternative "shortcut to Tier 3" if time is very tight:** instead of building attention from scratch, extract a small spatial patch (e.g., 5×5 grid cells) around each target point as additional input channels to your Tier 2 MLP — a cheap way to add spatial context without a full graph/attention architecture.

---

## 5. Evaluation — What to Actually Report

- **RMSE and MAE per depth level**, not just one blended number — subsurface reconstruction is typically most accurate near the surface and degrades with depth; showing this honestly is more credible than hiding it.
- **Correlation coefficient (R)** between predicted and actual Argo values on held-out test data.
- **A skill baseline**: compare against a trivial baseline (e.g., "climatological mean temperature for that location/month") — your model should clearly beat this, and showing the comparison proves your model is learning something real, not just regurgitating averages.
- Target ballpark from published literature: RMSE around 0.1–0.5°C near-surface, growing at depth — don't be alarmed if your first model is worse; iterate.

---

## 6. Visualization & Demo (this is what actually wins hackathon rounds)

1. **Depth-slice map**: pick one date, show a 2D map of predicted temperature at, say, 200m across your study region, with actual Argo point measurements overlaid as dots — a visually obvious "prediction matches reality" image.
2. **Vertical profile comparison**: pick one Argo float location, plot predicted vs. actual temperature-vs-depth curve side by side.
3. **Interactive front-end** (even simple): a web map (Leaflet/Plotly) where a user clicks a point and a depth slider shows the reconstructed profile at that location and date. This alone will make your demo stand out from teams showing only static plots.
4. **One-slide "so what"**: connect the result to a real use case — e.g., "this can feed into cyclone intensity models, since deep ocean heat content is a known predictor of rapid intensification."

---

## 7. Suggested Timeline (assuming ~5–6 day build window)

| Day | Focus |
|---|---|
| 1 | Register for Copernicus Marine + set up `argopy`; download a small test region/date range; confirm data loads in `xarray`. |
| 2 | Build the full preprocessing pipeline (QC filtering, depth binning, time-matching, train/val/test split by time). Train Tier 1 (XGBoost) baseline. |
| 3 | Build and train Tier 2 MLP; compare against Tier 1; start visualization scripts. |
| 4 | Refine model (hyperparameter tuning, add spatial patch context if time allows); build interactive demo front-end. |
| 5 | Polish visuals, prepare slides connecting to real-world impact, rehearse the demo, prepare answers for "how does this generalize / what are the limitations" questions. |
| 6 (buffer) | Fix whatever breaks — always assume something will. |

---

## 8. Common Pitfalls (avoid these — judges have seen them all)

- **Random train/test split** on time-series ocean data → inflated, unrealistic accuracy. Split by time.
- **Ignoring Argo QC flags** → noisy training labels, poor and inconsistent results.
- **Claiming "global" coverage** with a tiny regional dataset — scope your claims to match your data.
- **Only reporting one aggregate metric** — depth-wise breakdown is what shows real understanding.
- **No baseline comparison** — a number with nothing to compare against is meaningless to a judge.
- **Static plots only** — an interactive demo consistently scores better than the same result shown as screenshots.

---

## 9. Team Role Suggestions

- **1 person**: data engineering (Copernicus + Argo pipeline, preprocessing, QC).
- **1–2 people**: modeling (Tier 1 → Tier 2 → Tier 3 progression).
- **1 person**: visualization/front-end (map, profile viewer, slides).
- **1 person**: domain framing + presentation (read a couple of the cited papers below so you can answer "why does this work physically" convincingly — judges will ask).

---

## 10. Key References to Skim (for physical intuition, not to copy code from)

- Neural network approaches to inferring subsurface thermal structure from surface parameters — establishes the core surface→subsurface learning idea.
- Geographically weighted regression models for subsurface temperature anomaly — explains why SSH/altimetry is typically the most important input feature.
- Spatiotemporal graph attention network approaches — the state-of-the-art direction if you reach Tier 3.
- Satellite embedding / foundation-model approaches — a newer, lighter-weight alternative to hand-engineered features if you want to experiment.

---

**Bottom line:** get Tier 1 working by day 2 no matter what — it guarantees you have *something* to demo. Everything after that is upside.
