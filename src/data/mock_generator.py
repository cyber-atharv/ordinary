"""
High-Fidelity Physical Synthetic Ocean Data Generator for SIH26066
Generates realistic 3D volumetric ocean arrays for the North Indian Ocean
(Arabian Sea, Bay of Bengal, 5°N-30°N, 45°E-105°E at 0.25° daily resolution)
with mesoscale warm/cold eddies, monsoonal wind stress, and in-situ Argo float casts.
"""

import numpy as np
import os
import json
from typing import Dict, Any, Tuple
from src.data.sturm_liouville import STANDARD_DEPTHS, compute_standard_climatology_profile


def generate_north_indian_ocean_dataset(
    num_days: int = 10,
    lat_range: Tuple[float, float] = (5.0, 30.0),
    lon_range: Tuple[float, float] = (45.0, 105.0),
    resolution: float = 0.25,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates realistic daily multi-modal satellite surface tensors and 3D subsurface ground truth.
    
    Returns dictionary with:
        lats: array of shape [101]
        lons: array of shape [241]
        depths: array of shape [15]
        surface_features: array of shape [num_days, 7, 101, 241]
            (Channels: SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)
        ground_truth_3d: array of shape [num_days, 15, 101, 241]
        argo_floats: list of dicts representing in-situ casts for each day
    """
    np.random.seed(seed)
    
    lats = np.arange(lat_range[0], lat_range[1] + 1e-5, resolution, dtype=np.float32)
    lons = np.arange(lon_range[0], lon_range[1] + 1e-5, resolution, dtype=np.float32)
    num_lat = len(lats)
    num_lon = len(lons)
    depths = STANDARD_DEPTHS
    num_depths = len(depths)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 1. Land-Sea Mask: Indian Subcontinent & Arabian Peninsula
    # Approximate bounding polygon for land mass
    is_land = np.zeros((num_lat, num_lon), dtype=bool)
    # India triangle approx: lat > 8 & lon > 68 & lon < 88 & (lat > 8 + 0.9 * (lon-68) or lat > 22)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # North land mass (>24°N except Arabian sea up to 25°)
            if lat > 24.5 and 55.0 < lon < 95.0:
                is_land[i, j] = True
            # Indian subcontinent peninsular triangle
            elif 8.0 <= lat <= 24.5 and 72.0 <= lon <= 86.0:
                if lat > (8.0 + 1.2 * abs(lon - 79.0)):
                    is_land[i, j] = True
            # Arabian peninsula
            elif 13.0 <= lat and lon <= 58.0:
                is_land[i, j] = True
            # Myanmar/SE Asia coast
            elif 10.0 <= lat and lon >= 98.0:
                is_land[i, j] = True

    t_clim = compute_standard_climatology_profile(depths)
    
    surface_tensors = np.zeros((num_days, 7, num_lat, num_lon), dtype=np.float32)
    subsurface_3d = np.zeros((num_days, num_depths, num_lat, num_lon), dtype=np.float32)
    argo_float_catalog = []
    
    for d in range(num_days):
        day_phase = (d % 365) / 365.0 * 2 * np.pi
        
        # --- Physical Eddy Fields ---
        # 1. Arabian Sea Warm Pool (Anticyclonic Eddy at 12°N, 68°E)
        eddy1 = 0.25 * np.exp(-((lat_grid - 12.5)**2 + (lon_grid - 68.0)**2) / 8.0)
        # 2. Somali Upwelling Cold Eddy (Cyclonic Eddy at 10°N, 53°E)
        eddy2 = -0.35 * np.exp(-((lat_grid - 10.0)**2 + (lon_grid - 53.0)**2) / 10.0)
        # 3. Bay of Bengal Freshwater / Warm Gyre (15°N, 88°E)
        eddy3 = 0.20 * np.exp(-((lat_grid - 15.0)**2 + (lon_grid - 88.0)**2) / 12.0)
        
        sla_field = eddy1 + eddy2 + eddy3 + 0.05 * np.sin(lat_grid * 0.3 + day_phase)
        
        # SST: Climatological gradient (warmer in south ~29°C, cooler in north ~27°C) + Eddy anomalies
        base_sst = 29.5 - 0.08 * (lat_grid - 5.0) + 0.8 * (sla_field / 0.3)
        noise_sst = np.random.normal(0, 0.15, (num_lat, num_lon))
        sst_field = base_sst + noise_sst
        
        # SSS: Arabian sea is saline (~36 psu), Bay of Bengal is fresh (~32 psu due to river runoff)
        sss_field = 36.5 - 0.06 * (lon_grid - 50.0) - 0.5 * (sla_field / 0.3)
        
        # Surface Currents (Geostrophic approximation derived from SLA gradient)
        # u = -(g/f) d(SLA)/dy, v = (g/f) d(SLA)/dx
        d_sla_dy, d_sla_dx = np.gradient(sla_field, resolution * 111000, resolution * 111000)
        f_coriolis = 2 * 7.2921e-5 * np.sin(np.radians(np.maximum(lat_grid, 4.0)))
        u_curr = - (9.81 / f_coriolis) * d_sla_dy
        v_curr = (9.81 / f_coriolis) * d_sla_dx
        u_curr = np.clip(u_curr + 0.1 * np.cos(day_phase), -1.5, 1.5)
        v_curr = np.clip(v_curr + 0.1 * np.sin(day_phase), -1.5, 1.5)
        
        # Surface Winds (Southwest monsoonal flow in summer)
        u_wind = 6.0 + 3.0 * np.cos(day_phase) + np.random.normal(0, 0.5, (num_lat, num_lon))
        v_wind = 4.0 + 2.5 * np.sin(day_phase) + np.random.normal(0, 0.5, (num_lat, num_lon))
        
        # Apply Land Mask
        sst_field[is_land] = np.nan
        sss_field[is_land] = np.nan
        sla_field[is_land] = np.nan
        u_curr[is_land] = 0.0
        v_curr[is_land] = 0.0
        u_wind[is_land] = 0.0
        v_wind[is_land] = 0.0
        
        # Store 7-channel surface tensor
        surface_tensors[d, 0] = np.nan_to_num(sst_field, nan=0.0)
        surface_tensors[d, 1] = np.nan_to_num(sss_field, nan=35.0)
        surface_tensors[d, 2] = np.nan_to_num(sla_field, nan=0.0)
        surface_tensors[d, 3] = u_curr
        surface_tensors[d, 4] = v_curr
        surface_tensors[d, 5] = u_wind
        surface_tensors[d, 6] = v_wind
        
        # --- 3D Subsurface Temperature Field Reconstruction (Ground Truth) ---
        # Baroclinic coupling: Positive SLA (warm eddy) deepens thermocline (warmer at 100-200m)
        # Negative SLA (upwelling) shoals thermocline (colder at 50-150m)
        for k, z in enumerate(depths):
            thermocline_depth_response = np.exp(-((z - 120.0)**2) / (2 * 65.0**2))
            vertical_eddy_t = (sla_field / 0.15) * 2.8 * thermocline_depth_response
            
            deep_attenuation = np.exp(-z / 350.0)
            layer_t = t_clim[k] + (sst_field - 28.5) * deep_attenuation + vertical_eddy_t
            layer_t[is_land] = np.nan
            subsurface_3d[d, k] = np.nan_to_num(layer_t, nan=0.0)
            
        # --- Generate 15 Realistic Argo Float In-Situ Profiles for Day d ---
        num_floats = 15
        for f in range(num_floats):
            # Pick a valid ocean coordinate
            for _ in range(50):
                f_lat_idx = np.random.randint(5, num_lat - 5)
                f_lon_idx = np.random.randint(5, num_lon - 5)
                if not is_land[f_lat_idx, f_lon_idx]:
                    break
                    
            f_lat = float(lats[f_lat_idx])
            f_lon = float(lons[f_lon_idx])
            f_profile = subsurface_3d[d, :, f_lat_idx, f_lon_idx].tolist()
            
            argo_float_catalog.append({
                "day_index": d,
                "float_id": f"ARGO_IND_{1000 + f}",
                "latitude": round(f_lat, 2),
                "longitude": round(f_lon, 2),
                "qc_flag": 1,
                "temperatures": [round(t, 2) for t in f_profile],
                "depths": depths.tolist()
            })

    return {
        "lats": lats,
        "lons": lons,
        "depths": depths,
        "is_land": is_land,
        "surface_features": surface_tensors,
        "ground_truth_3d": subsurface_3d,
        "argo_floats": argo_float_catalog
    }


def save_mock_dataset(output_dir: str = "data/synthetic"):
    """Generates and saves the synthetic arrays for offline testing."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Data Pipeline] Generating High-Fidelity North Indian Ocean dataset...")
    data = generate_north_indian_ocean_dataset(num_days=5)
    
    np.save(os.path.join(output_dir, "surface_features.npy"), data["surface_features"])
    np.save(os.path.join(output_dir, "ground_truth_3d.npy"), data["ground_truth_3d"])
    np.save(os.path.join(output_dir, "lats.npy"), data["lats"])
    np.save(os.path.join(output_dir, "lons.npy"), data["lons"])
    np.save(os.path.join(output_dir, "is_land.npy"), data["is_land"])
    
    with open(os.path.join(output_dir, "argo_floats.json"), "w") as f:
        json.dump(data["argo_floats"], f, indent=2)
        
    print(f"[Data Pipeline] Successfully generated and saved dataset in '{output_dir}':")
    print(f"  • Surface Tensors: {data['surface_features'].shape} (SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)")
    print(f"  • 3D Subsurface Grids: {data['ground_truth_3d'].shape} across 15 standard depths (0-1000m)")
    print(f"  • Argo Float Casts: {len(data['argo_floats'])} in-situ profiles with QC=1")


if __name__ == "__main__":
    save_mock_dataset()
