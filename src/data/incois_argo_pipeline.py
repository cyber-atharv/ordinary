"""
INCOIS LAS Gridded ARGO Data Pipeline

Retrieves gridded Argo temperature profiles from the INCOIS Live Access Server (LAS)
for independent validation of the subsurface reconstruction model.

The INCOIS LAS provides quality-controlled gridded Argo products covering the
Indian Ocean basin, which serve as ground truth for evaluating the reconstructed
3D temperature fields at the 15 standard depth levels.
"""

import os
import numpy as np
import xarray as xr
from typing import Optional, Tuple


STANDARD_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

# INCOIS LAS base URL for gridded Argo products
INCOIS_LAS_BASE_URL = "https://las.incois.gov.in"


def fetch_incois_gridded_argo(
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    lat_range: Tuple[float, float] = (5.0, 30.0),
    lon_range: Tuple[float, float] = (45.0, 105.0),
) -> str:
    """
    Fetches gridded Argo temperature data from the INCOIS Live Access Server.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        output_dir: Directory to store downloaded data
        lat_range: Latitude bounds (South, North)
        lon_range: Longitude bounds (West, East)

    Returns:
        Path to the downloaded/cached NetCDF file

    Note:
        If INCOIS LAS is unavailable, falls back to standard argopy GDAC retrieval.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"incois_argo_{start_date}_{end_date}.nc")

    if os.path.exists(output_file):
        print(f"INCOIS Argo data already cached: {output_file}")
        return output_file

    # Attempt INCOIS LAS OPeNDAP access
    try:
        print(f"Fetching gridded ARGO from INCOIS LAS: {start_date} to {end_date}")
        # INCOIS LAS provides OPeNDAP endpoints for gridded Argo fields
        # Adjust the URL template based on the specific LAS product catalog
        ds = xr.open_dataset(
            f"{INCOIS_LAS_BASE_URL}/opendap/argo_gridded_temp",
            engine="netcdf4",
        )
        ds_subset = ds.sel(
            latitude=slice(lat_range[0], lat_range[1]),
            longitude=slice(lon_range[0], lon_range[1]),
            time=slice(start_date, end_date),
        )
        ds_subset.to_netcdf(output_file)
        print(f"Saved INCOIS gridded ARGO to: {output_file}")
        return output_file

    except Exception as e:
        print(f"INCOIS LAS access failed ({e}), falling back to argopy GDAC")
        return _fallback_argopy_gridded(start_date, end_date, output_dir, lat_range, lon_range)


def _fallback_argopy_gridded(start_date, end_date, output_dir, lat_range, lon_range):
    """Fallback: retrieve Argo profiles via argopy and create a gridded product."""
    try:
        import argopy
        fetcher = argopy.DataFetcher(src="gdac").region(
            [lon_range[0], lon_range[1], lat_range[0], lat_range[1], 0, 1000],
            [start_date, end_date],
        )
        ds = fetcher.to_xarray()
        output_file = os.path.join(output_dir, f"argopy_fallback_{start_date}_{end_date}.nc")
        ds.to_netcdf(output_file)
        print(f"Saved argopy fallback data to: {output_file}")
        return output_file
    except Exception as e:
        print(f"argopy fallback also failed: {e}")
        return None


def load_validation_profiles(
    argo_path: str,
    target_depths: np.ndarray = STANDARD_DEPTHS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads gridded Argo data and interpolates to standard depth levels for validation.

    Returns:
        lats: Latitude array
        lons: Longitude array
        depths: Standard depth levels
        temperature: Array of shape [Time, Depths, Lat, Lon]
    """
    ds = xr.open_dataset(argo_path)

    # Identify temperature variable name (varies across ARGO products)
    temp_var = None
    for candidate in ["TEMP", "temperature", "thetao", "temp"]:
        if candidate in ds.data_vars:
            temp_var = candidate
            break

    if temp_var is None:
        raise ValueError(f"No temperature variable found in dataset: {list(ds.data_vars)}")

    # Interpolate to standard depths if needed
    if "depth" in ds.dims:
        ds = ds.interp(depth=target_depths, method="linear")

    temperature = ds[temp_var].values.astype(np.float32)
    lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
    lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values

    print(f"Loaded validation profiles: {temperature.shape}")
    return lats, lons, target_depths, temperature
