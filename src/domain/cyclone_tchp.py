"""
Tropical Cyclone Heat Potential (TCHP) & Isotherm Depth Analytics
Calculates:
1. D26: Depth of the 26°C Isotherm (meters)
2. TCHP: Tropical Cyclone Heat Potential (kJ/cm^2 or MJ/m^2)
    TCHP = rho * Cp * integral_{0}^{D26} (T(z) - 26) dz
3. Mixed Layer Depth (MLD) using threshold criteria (Delta T = 0.2°C from surface)
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple
from src.data.sturm_liouville import STANDARD_DEPTHS


def compute_tchp_and_d26_numpy(
    t_3d: np.ndarray, 
    depths: np.ndarray = STANDARD_DEPTHS,
    rho: float = 1025.0,  # Seawater density (kg/m^3)
    cp: float = 3985.0    # Specific heat capacity (J / kg * K)
) -> Dict[str, np.ndarray]:
    """
    Computes TCHP (kJ/cm^2) and D26 (m) across 3D temperature grids.
    Args:
        t_3d: Array of shape [15, H, W] or [15] (Temperature at standard depths in °C)
        depths: Array of shape [15] (Depths in meters)
    Returns:
        Dictionary with 'tchp_kj_cm2', 'd26_m', and 'mld_m'
    """
    if t_3d.ndim == 1:
        # Single 1D profile
        t_prof = t_3d
        # 1. Compute D26 (26°C isotherm depth)
        if t_prof[0] < 26.0:
            d26 = 0.0
            tchp = 0.0
        else:
            # Interpolate depth where T = 26.0°C
            idx_below = np.where(t_prof < 26.0)[0]
            if len(idx_below) == 0:
                d26 = float(depths[-1])
            else:
                first_below = idx_below[0]
                prev = first_below - 1
                t1, t2 = t_prof[prev], t_prof[first_below]
                z1, z2 = depths[prev], depths[first_below]
                if abs(t2 - t1) > 1e-6:
                    d26 = z1 + (26.0 - t1) * (z2 - z1) / (t2 - t1)
                else:
                    d26 = z1
                    
            # 2. Integrate Heat Content above 26°C
            # TCHP = rho * Cp * sum ( (T_mid - 26) * delta_z ) in J/m^2 -> / 1e7 for kJ/cm^2
            integral_j_m2 = 0.0
            for i in range(len(depths) - 1):
                z_top, z_bot = depths[i], depths[i+1]
                if z_top >= d26:
                    break
                z_bot_eff = min(z_bot, d26)
                dz = z_bot_eff - z_top
                t_mid = 0.5 * (t_prof[i] + min(t_prof[i+1], max(t_prof[i], 26.0)))
                excess_t = max(0.0, t_mid - 26.0)
                integral_j_m2 += rho * cp * excess_t * dz
                
            tchp = integral_j_m2 / 1.0e7  # Convert J/m^2 to kJ/cm^2 (standard IMD/NOAA units)

        # 3. Compute Mixed Layer Depth (MLD) (Delta T = 0.2°C from surface)
        mld = 30.0
        for i in range(1, len(depths)):
            if (t_prof[0] - t_prof[i]) >= 0.2:
                mld = float(depths[i])
                break

        return {
            "tchp_kj_cm2": float(round(tchp, 2)),
            "d26_m": float(round(d26, 1)),
            "mld_m": float(round(mld, 1))
        }
        
    elif t_3d.ndim == 3:
        # Full 3D Grid: [15, H, W]
        num_z, h, w = t_3d.shape
        tchp_grid = np.zeros((h, w), dtype=np.float32)
        d26_grid = np.zeros((h, w), dtype=np.float32)
        mld_grid = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                if not np.isnan(t_3d[0, i, j]) and t_3d[0, i, j] > 0.0:
                    res = compute_tchp_and_d26_numpy(t_3d[:, i, j], depths, rho, cp)
                    tchp_grid[i, j] = res["tchp_kj_cm2"]
                    d26_grid[i, j] = res["d26_m"]
                    mld_grid[i, j] = res["mld_m"]
                else:
                    tchp_grid[i, j] = np.nan
                    d26_grid[i, j] = np.nan
                    mld_grid[i, j] = np.nan
                    
        return {
            "tchp_kj_cm2": tchp_grid,
            "d26_m": d26_grid,
            "mld_m": mld_grid
        }
    else:
        raise ValueError(f"Unsupported array shape: {t_3d.shape}")
