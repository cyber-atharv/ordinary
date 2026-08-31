"""
Marine Heatwave (MHW) & Coral Reef Ecological Impact Engine
Tracks Subsurface Marine Heatwaves (SMHW) and depth-penetrating thermal stress.
"""

import numpy as np
from typing import Dict, Any
from src.data.sturm_liouville import STANDARD_DEPTHS, compute_standard_climatology_profile


def detect_subsurface_marine_heatwaves(
    temperature: np.ndarray,
    depths: np.ndarray = STANDARD_DEPTHS
) -> Dict[str, Any]:
    """
    Detects depth-penetrating marine heatwaves where temperature exceeds
    the 90th percentile climatological threshold.
    """
    t_clim = compute_standard_climatology_profile(depths)
    anomalies = temperature - t_clim
    
    # 90th percentile threshold approximation: Delta T > +1.2°C
    mhw_mask = anomalies > 1.2
    
    max_anomaly = float(np.max(anomalies))
    max_anomaly_depth = float(depths[int(np.argmax(anomalies))])
    
    # Penetration depth of heatwave
    penetration_depth = 0.0
    for i in range(len(depths)):
        if mhw_mask[i]:
            penetration_depth = float(depths[i])
            
    # Severity classification (Hobday et al. 2018)
    if max_anomaly > 3.0:
        category = "Category IV: Extreme Subsurface Heatwave"
    elif max_anomaly > 2.0:
        category = "Category III: Severe Subsurface Heatwave"
    elif max_anomaly > 1.2:
        category = "Category II: Moderate Heatwave"
    else:
        category = "Normal Thermal State"

    return {
        "mhw_category": category,
        "max_thermal_anomaly_degC": round(max_anomaly, 2),
        "peak_anomaly_depth_m": max_anomaly_depth,
        "thermal_penetration_depth_m": penetration_depth,
        "coral_reef_bleaching_risk": "HIGH" if (max_anomaly > 1.5 and penetration_depth >= 30.0) else "LOW",
        "ecological_impact_summary": "Thermal stress penetrating below 30m causes deep bleaching on mesophotic coral reefs in Lakshadweep and Andaman Sea."
    }
