"""
Potential Fishing Zone (PFZ) & Thermocline Upwelling Advisory Module

Implements physical oceanographic algorithms for INCOIS's flagship daily PFZ advisory:
1. D20 Isotherm Depth: Locates the 20 deg C isotherm depth (indicative of upper-ocean thermocline).
2. Vertical Temperature Gradient (dT/dz): Quantifies thermocline strength.
3. Coastal / Open-Ocean Upwelling Index: Identifies upward nutrient pumping zones.
4. Fishing Potential Score: High / Moderate / Low aggregation advisories for pelagic fish shoals.
"""

import numpy as np
from typing import Dict, Any, List
from src.data.sturm_liouville import STANDARD_DEPTHS


def compute_d20_isotherm_depth(
    temperature_profile: np.ndarray,
    depths: np.ndarray = STANDARD_DEPTHS
) -> float:
    """
    Computes the depth of the 20 deg C isotherm (D20) via linear vertical interpolation.
    D20 is the primary oceanographic proxy for the thermocline depth in the tropical Indian Ocean.
    """
    temp = np.asarray(temperature_profile, dtype=np.float32)
    z = np.asarray(depths, dtype=np.float32)

    if temp[0] < 20.0:
        return float(z[0])
    if temp[-1] > 20.0:
        return float(z[-1])

    for i in range(len(temp) - 1):
        if temp[i] >= 20.0 and temp[i + 1] <= 20.0:
            fraction = (temp[i] - 20.0) / (temp[i] - temp[i + 1] + 1e-6)
            d20 = z[i] + fraction * (z[i + 1] - z[i])
            return float(np.clip(d20, z[0], z[-1]))

    return float(z[-1])


def analyze_pfz_and_upwelling(
    temperature_profile: np.ndarray,
    lat: float,
    lon: float,
    depths: np.ndarray = STANDARD_DEPTHS
) -> Dict[str, Any]:
    """
    Computes comprehensive Potential Fishing Zone (PFZ) indicators.

    Args:
        temperature_profile: 15-depth temperature array (deg C)
        lat: Target latitude
        lon: Target longitude
        depths: 15 standard oceanographic depth levels

    Returns:
        Dictionary with PFZ potential, D20 depth, upwelling intensity, and advisory notes.
    """
    temp = np.asarray(temperature_profile, dtype=np.float32)
    d20 = compute_d20_isotherm_depth(temp, depths)

    # Thermocline gradient: max temperature drop per meter in upper 200m
    upper_mask = depths <= 200.0
    upper_temps = temp[upper_mask]
    upper_depths = depths[upper_mask]
    
    dt = upper_temps[:-1] - upper_temps[1:]
    dz = upper_depths[1:] - upper_depths[:-1]
    gradients = dt / (dz + 1e-6)
    max_gradient = float(np.max(gradients)) if len(gradients) > 0 else 0.05

    # Upwelling classification based on D20 depth and thermocline shallowness
    # Shallow D20 (<65m) indicates strong upward lifting of cold, nutrient-rich deep water
    if d20 <= 55.0:
        upwelling_state = "Active Upwelling (Strong Divergence)"
        pfz_potential = "High (Optimal Fish Aggregation)"
        score = 88.5
        recommended_fishing_depth_m = round(d20 - 10.0, 1)
        advisory = (
            f"Strong subsurface upwelling detected at ({lat}N, {lon}E). D20 isotherm is shallow ({d20:.1f}m). "
            f"Nutrient pumping promotes high chlorophyll and pelagic fish shoaling (Tuna, Mackerel) at ~{recommended_fishing_depth_m}m depth."
        )
    elif d20 <= 90.0:
        upwelling_state = "Moderate Stratification"
        pfz_potential = "Moderate"
        score = 62.0
        recommended_fishing_depth_m = round(d20 - 15.0, 1)
        advisory = (
            f"Moderate thermocline depth at ({lat}N, {lon}E) with D20 at {d20:.1f}m. "
            f"Scattered pelagic aggregation expected near {recommended_fishing_depth_m}m depth."
        )
    else:
        upwelling_state = "Deep Downwelling / Strong Thermal Barrier"
        pfz_potential = "Low (Deep Thermocline)"
        score = 34.0
        recommended_fishing_depth_m = round(min(d20, 100.0), 1)
        advisory = (
            f"Deep thermocline (D20 = {d20:.1f}m) indicates downwelling warm pool. "
            f"Low vertical nutrient transport at this location."
        )

    return {
        "d20_isotherm_depth_m": round(d20, 1),
        "max_thermocline_gradient_degC_m": round(max_gradient, 3),
        "upwelling_status": upwelling_state,
        "pfz_potential_category": pfz_potential,
        "pfz_suitability_score": round(score, 1),
        "recommended_gear_depth_m": recommended_fishing_depth_m,
        "incois_advisory_text": advisory
    }
