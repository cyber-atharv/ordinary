"""
In-Situ Argo Float Data Fetching & Quality Control Pipeline
Uses argopy to fetch high-precision temperature casts across the North Indian Ocean.
Applies strict QC Flag = 1 filtering and PCHIP vertical interpolation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from src.data.sturm_liouville import STANDARD_DEPTHS

try:
    import argopy
    ARGOPY_AVAILABLE = True
except ImportError:
    ARGOPY_AVAILABLE = False


def fetch_argo_profiles_region(
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31"
) -> Optional[Any]:
    """
    Fetches real-world Argo profiling float casts via argopy GDAC API.
    """
    if not ARGOPY_AVAILABLE:
        print("[ArgoPipeline] 'argopy' is not installed. Using local cache/mock profiles.")
        return None
        
    print(f"[ArgoPipeline] Fetching Argo profiles for region [{lon_min}, {lon_max}, {lat_min}, {lat_max}] from {start_date} to {end_date}...")
    try:
        from argopy import DataFetcher as ArgoDataFetcher
        fetcher = ArgoDataFetcher(src='erddap', mode='standard')
        ds = fetcher.region([lon_min, lon_max, lat_min, lat_max, 0, 1000, start_date, end_date]).to_xarray()
        
        # Apply strict QC filtering (QC flag = 1 indicates good quality measurements)
        if 'TEMP_QC' in ds:
            ds = ds.where(ds.TEMP_QC == 1, drop=True)
            
        print(f"[ArgoPipeline] Successfully retrieved {len(ds.N_PROF)} valid QC=1 Argo profiles.")
        return ds
    except Exception as e:
        print(f"[ArgoPipeline] Could not fetch remote Argo data (check network): {e}")
        return None


def interpolate_profile_to_standard_depths(
    raw_depths: np.ndarray,
    raw_temperatures: np.ndarray,
    target_depths: np.ndarray = STANDARD_DEPTHS
) -> np.ndarray:
    """
    Uses shape-preserving monotonic PCHIP interpolation to interpolate
    irregular pressure/depth sensor readings onto standard depths.
    """
    from scipy.interpolate import PchipInterpolator
    
    # Filter NaNs
    valid = (~np.isnan(raw_depths)) & (~np.isnan(raw_temperatures))
    if np.sum(valid) < 4:
        # Fallback to linear
        return np.interp(target_depths, raw_depths[valid], raw_temperatures[valid], left=np.nan, right=np.nan)
        
    pchip = PchipInterpolator(raw_depths[valid], raw_temperatures[valid], extrapolate=True)
    return pchip(target_depths).astype(np.float32)
