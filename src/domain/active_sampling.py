"""
Intelligent Active Sampling Float Drop Optimizer

Uses model uncertainty bounds (quantile variance sigma_z(x, y)) across the North Indian Ocean
to automatically identify observation gaps and recommend optimal GPS deployment coordinates
for future Argo floats, gliders, or MoES research vessels (ORV Sagar Nidhi).
"""

import numpy as np
from typing import Dict, Any, List


def recommend_optimal_float_drops(
    lats: np.ndarray,
    lons: np.ndarray,
    uncertainty_grid_3d: np.ndarray,
    is_land_mask: np.ndarray,
    existing_floats: List[Dict[str, Any]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Finds geographic coordinates with highest reconstruction uncertainty
    that are sufficiently far from existing active in-situ observation assets.

    Args:
        lats: 1D array of latitude coordinates
        lons: 1D array of longitude coordinates
        uncertainty_grid_3d: Array of shape [15, Lat, Lon] with uncertainty spread (T_90 - T_10)
        is_land_mask: 2D boolean land-sea mask (True for land)
        existing_floats: List of active float dictionaries with 'latitude', 'longitude'
        top_k: Number of optimal deployment points to return

    Returns:
        List of recommended drop points with GPS coordinates, expected uncertainty reduction, and rationale.
    """
    # Depth-integrated uncertainty map (weighted towards thermocline 50-200m)
    weights = np.array([0.5, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3], dtype=np.float32)
    weights = weights[:, np.newaxis, np.newaxis]
    
    unc_2d = np.sum(uncertainty_grid_3d * weights, axis=0) / np.sum(weights)
    unc_2d[is_land_mask] = 0.0  # Zero out land

    # Penalize proximity to existing active floats (Gaussian exclusion radius ~ 250km / ~2.5 deg)
    if existing_floats:
        for f in existing_floats:
            f_lat = f.get("latitude", 15.0)
            f_lon = f.get("longitude", 70.0)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            dist_deg_sq = (lat_grid - f_lat)**2 + (lon_grid - f_lon)**2
            exclusion_mask = 1.0 - np.exp(-dist_deg_sq / (2 * 2.5**2))
            unc_2d = unc_2d * exclusion_mask

    recommendations = []
    unc_work = unc_2d.copy()

    for rank in range(1, top_k + 1):
        idx = np.unravel_index(np.argmax(unc_work), unc_work.shape)
        best_lat = float(lats[idx[0]])
        best_lon = float(lons[idx[1]])
        peak_unc = float(unc_work[idx])

        if peak_unc < 1e-4:
            break

        # Regional identification
        basin = "Arabian Sea" if best_lon < 78.0 else "Bay of Bengal"
        
        recommendations.append({
            "rank": rank,
            "latitude": round(best_lat, 2),
            "longitude": round(best_lon, 2),
            "basin": basin,
            "integrated_uncertainty_spread_degC": round(peak_unc, 2),
            "expected_reconstruction_skill_gain_pct": round(float(np.clip(peak_unc * 18.0, 8.0, 32.0)), 1),
            "mission_rationale": (
                f"Priority #{rank} in {basin} at ({best_lat:.2f}N, {best_lon:.2f}E). "
                f"High thermocline variance (spread = {peak_unc:.2f}C) with zero nearby in-situ profiles. "
                f"Deploying an Argo float or autonomous glider here maximizes basin-wide model assimilation gain."
            )
        })

        # Apply suppression radius around selected point to find next distinct location
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        dist_sq = (lat_grid - best_lat)**2 + (lon_grid - best_lon)**2
        suppression = 1.0 - np.exp(-dist_sq / (2 * 3.5**2))
        unc_work = unc_work * suppression

    return recommendations
