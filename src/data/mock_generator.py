"""
Realistic Ocean Data Generator for SIH26066
Created to generate realistic 3D ocean data across the North Indian Ocean
(Arabian Sea and Bay of Bengal, 5°N-30°N, 45°E-105°E on a 0.25° daily grid).

This lets you test the full AI pipeline offline without needing to download
gigabytes of raw satellite files from NASA or Copernicus servers.
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
    Builds a simulated daily dataset with 2D satellite surface observations
    and corresponding 3D underwater temperature profiles (ground truth).
    """
    np.random.seed(seed)
    
    # Create the latitude and longitude coordinate axes
    lats = np.arange(lat_range[0], lat_range[1] + 1e-5, resolution, dtype=np.float32)
    lons = np.arange(lon_range[0], lon_range[1] + 1e-5, resolution, dtype=np.float32)
    num_lat = len(lats)
    num_lon = len(lons)
    depths = STANDARD_DEPTHS
    num_depths = len(depths)
    
    # 2D coordinate grids for fast vectorized math
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Step 1: Define coastal polygons so we can keep the land clean and strictly color the sea
    LAND_POLYGONS = [
        # Indian Subcontinent & Northern Eurasian landmass
        [
            [24.8, 61.5], [25.3, 63.5], [25.1, 64.5], [25.3, 66.5],
            [24.8, 67.0], [24.0, 67.8], [23.7, 68.2], [23.2, 68.5],
            [22.8, 69.1], [22.8, 70.3], [22.5, 69.5], [22.2, 68.9],
            [21.5, 69.6], [20.8, 70.4], [20.7, 71.0], [21.0, 72.1],
            [22.2, 72.6], [21.7, 72.7], [21.1, 72.7], [20.4, 72.8],
            [19.0, 72.8], [18.0, 73.0], [17.0, 73.3], [16.0, 73.5],
            [15.4, 73.8], [14.5, 74.3], [13.3, 74.7], [12.9, 74.8],
            [12.0, 75.2], [11.2, 75.8], [10.0, 76.2], [9.5, 76.3],
            [8.8, 76.6],  [8.5, 76.9],  [8.08, 77.55],
            [8.8, 78.1],  [9.28, 79.3], [9.8, 79.0],  [10.3, 79.85],
            [10.8, 79.85], [11.9, 79.8], [13.08, 80.27], [14.0, 80.1],
            [15.5, 80.2], [16.0, 80.8], [16.9, 82.2], [17.7, 83.3],
            [18.5, 84.3], [19.3, 85.0], [19.8, 85.8], [20.3, 86.7],
            [21.5, 87.0], [21.6, 88.0], [21.7, 89.0], [22.0, 90.5],
            [22.3, 91.8], [21.4, 92.0], [20.5, 92.4],
            [20.5, 93.0], [32.0, 93.0], [32.0, 61.5], [24.8, 61.5]
        ],
        # Sri Lanka
        [
            [9.8, 80.2], [9.3, 80.0], [8.6, 79.8], [8.0, 79.7],
            [7.0, 79.8], [6.0, 80.2], [5.9, 80.5], [6.2, 81.3],
            [7.0, 81.9], [7.7, 81.7], [8.6, 81.2], [9.3, 80.6],
            [9.8, 80.2]
        ],
        # Arabian Peninsula
        [
            [12.6, 43.4], [12.8, 45.0], [13.5, 46.5], [14.0, 47.0],
            [14.3, 48.5], [15.0, 50.5], [16.0, 52.0], [16.6, 53.0],
            [17.0, 54.1], [18.0, 56.0], [19.6, 57.7], [20.5, 58.8],
            [22.5, 59.8], [23.6, 58.6], [24.5, 56.8], [26.2, 56.4],
            [32.0, 56.4], [32.0, 43.0], [12.0, 43.0], [12.6, 43.4]
        ],
        # Iran Coast
        [
            [24.8, 61.5], [25.4, 60.5], [25.4, 59.0], [27.1, 56.5],
            [32.0, 56.5], [32.0, 61.5], [24.8, 61.5]
        ],
        # Horn of Africa
        [
            [12.0, 43.0], [11.5, 43.1], [10.5, 45.0], [11.0, 47.0],
            [11.5, 50.0], [11.8, 51.3], [10.4, 51.3], [7.9, 49.8],
            [5.3, 48.5],  [4.0, 47.0],  [4.0, 43.0],  [12.0, 43.0]
        ],
        # Indochina (Myanmar, Thailand, Malaysia)
        [
            [20.5, 92.4], [20.0, 92.8], [18.5, 93.8], [16.0, 94.2],
            [15.8, 95.0], [16.0, 96.0], [16.5, 97.0], [14.0, 98.0],
            [12.0, 98.5], [9.8, 98.5],  [8.0, 98.3],  [6.0, 99.8],
            [5.0, 100.3], [4.0, 100.5], [4.0, 106.0], [32.0, 106.0],
            [32.0, 92.4], [20.5, 92.4]
        ],
        # Sumatra
        [
            [5.6, 95.3], [4.5, 96.0], [3.5, 97.0], [2.0, 98.0],
            [2.0, 95.0], [5.6, 95.3]
        ]
    ]

    def point_in_polygon(lat_val, lon_val, poly):
        inside = False
        n = len(poly)
        p1y, p1x = poly[0]
        for idx in range(n + 1):
            p2y, p2x = poly[idx % n]
            if lat_val > min(p1y, p2y):
                if lat_val <= max(p1y, p2y):
                    if lon_val <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (lat_val - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or lon_val <= xinters:
                            inside = not inside
            p1y, p1x = p2y, p2x
        return inside

    is_land = np.zeros((num_lat, num_lon), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if lat >= 25.5:
                is_land[i, j] = True
            else:
                for poly in LAND_POLYGONS:
                    if point_in_polygon(lat, lon, poly):
                        is_land[i, j] = True
                        break

    t_clim = compute_standard_climatology_profile(depths)
    
    surface_tensors = np.zeros((num_days, 7, num_lat, num_lon), dtype=np.float32)
    subsurface_3d = np.zeros((num_days, num_depths, num_lat, num_lon), dtype=np.float32)
    argo_float_catalog = []
    
    for d in range(num_days):
        # Calculate the day of the year in radians for gentle seasonal changes
        day_phase = (d % 365) / 365.0 * 2 * np.pi
        
        # Step 2: Simulate natural swirling ocean eddies (whirlpools)
        # 1. Warm core eddy in the Arabian Sea Mini Warm Pool (12.5°N, 68°E)
        eddy1 = 0.25 * np.exp(-((lat_grid - 12.5)**2 + (lon_grid - 68.0)**2) / 8.0)
        # 2. Cold upwelling eddy along the Somali coast (10°N, 53°E)
        eddy2 = -0.35 * np.exp(-((lat_grid - 10.0)**2 + (lon_grid - 53.0)**2) / 10.0)
        # 3. Warm fresh gyre in the Bay of Bengal (15°N, 88°E)
        eddy3 = 0.20 * np.exp(-((lat_grid - 15.0)**2 + (lon_grid - 88.0)**2) / 12.0)
        
        # Sea surface height anomaly (small bumps and dips in water elevation)
        sla_field = eddy1 + eddy2 + eddy3 + 0.05 * np.sin(lat_grid * 0.3 + day_phase)
        
        # Step 3: Sea surface temperature (warmer near equator ~29.5°C, cooler up north ~27°C)
        base_sst = 29.5 - 0.08 * (lat_grid - 5.0) + 0.8 * (sla_field / 0.3)
        noise_sst = np.random.normal(0, 0.15, (num_lat, num_lon))
        sst_field = base_sst + noise_sst
        
        # Step 4: Salinity (Arabian Sea is extra salty ~36.5 PSU due to evaporation, BoB is fresher ~32 PSU)
        sss_field = 36.5 - 0.06 * (lon_grid - 50.0) - 0.5 * (sla_field / 0.3)
        
        # Step 5: Surface water currents derived from water height slopes (geostrophic balance)
        d_sla_dy, d_sla_dx = np.gradient(sla_field, resolution * 111000, resolution * 111000)
        f_coriolis = 2 * 7.2921e-5 * np.sin(np.radians(np.maximum(lat_grid, 4.0)))
        u_curr = - (9.81 / f_coriolis) * d_sla_dy
        v_curr = (9.81 / f_coriolis) * d_sla_dx
        u_curr = np.clip(u_curr + 0.1 * np.cos(day_phase), -1.5, 1.5)
        v_curr = np.clip(v_curr + 0.1 * np.sin(day_phase), -1.5, 1.5)
        
        # Step 6: Southwest summer monsoon wind flow
        u_wind = 6.0 + 3.0 * np.cos(day_phase) + np.random.normal(0, 0.5, (num_lat, num_lon))
        v_wind = 4.0 + 2.5 * np.sin(day_phase) + np.random.normal(0, 0.5, (num_lat, num_lon))
        
        # Zero out any land areas so data is strictly for ocean water
        sst_field[is_land] = np.nan
        sss_field[is_land] = np.nan
        sla_field[is_land] = np.nan
        u_curr[is_land] = 0.0
        v_curr[is_land] = 0.0
        u_wind[is_land] = 0.0
        v_wind[is_land] = 0.0
        
        # Pack the 7 surface observations into our daily feature array
        surface_tensors[d, 0] = np.nan_to_num(sst_field, nan=0.0)
        surface_tensors[d, 1] = np.nan_to_num(sss_field, nan=35.0)
        surface_tensors[d, 2] = np.nan_to_num(sla_field, nan=0.0)
        surface_tensors[d, 3] = u_curr
        surface_tensors[d, 4] = v_curr
        surface_tensors[d, 5] = u_wind
        surface_tensors[d, 6] = v_wind
        
        # Step 7: Build the 3D underwater temperature profile across all 15 depths
        # Warm eddies push the warm layer deeper down, while cold upwelling brings cold water up
        for k, z in enumerate(depths):
            thermocline_depth_response = np.exp(-((z - 120.0)**2) / (2 * 65.0**2))
            vertical_eddy_t = (sla_field / 0.15) * 2.8 * thermocline_depth_response
            
            deep_attenuation = np.exp(-z / 350.0)
            layer_t = t_clim[k] + (sst_field - 28.5) * deep_attenuation + vertical_eddy_t
            layer_t[is_land] = np.nan
            subsurface_3d[d, k] = np.nan_to_num(layer_t, nan=0.0)
            
        # Step 8: Simulate real in-situ Argo floats drifting in the sea on this day
        num_floats = 15
        for f in range(num_floats):
            # Pick a random point in the water, making sure it isn't on land
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
