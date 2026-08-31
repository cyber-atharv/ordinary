"""
GLORYS12V1 Global Ocean Reanalysis Data Loader

Downloads and preprocesses GLORYS12V1 subsurface temperature fields from Copernicus
Marine Service for use as training targets. GLORYS provides 3D ocean temperature
reanalysis at 1/12 deg resolution, which we regrid to the standard 0.25 deg grid.

Reference: https://doi.org/10.48670/moi-00021
"""

import os
import numpy as np
import xarray as xr
from typing import Optional, Tuple


# Standard oceanographic depth levels (meters) as specified by INCOIS SIH26066
STANDARD_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

# North Indian Ocean bounding box
LAT_RANGE = (5.0, 30.0)
LON_RANGE = (45.0, 105.0)
TARGET_RESOLUTION = 0.25  # degrees


def download_glorys_temperature(
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """
    Downloads GLORYS12V1 daily ocean temperature for the North Indian Ocean.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        output_dir: Directory to save downloaded NetCDF files
        username: Copernicus Marine username (or set COPERNICUSMARINE_USER env var)
        password: Copernicus Marine password (or set COPERNICUSMARINE_PASSWORD env var)

    Returns:
        Path to the downloaded NetCDF file
    """
    try:
        import copernicusmarine
    except ImportError:
        raise ImportError("Install copernicusmarine: pip install copernicusmarine")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"glorys12v1_temp_{start_date}_{end_date}.nc")

    if os.path.exists(output_file):
        print(f"GLORYS file already exists: {output_file}")
        return output_file

    # Copernicus Marine dataset ID for GLORYS12V1
    dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    print(f"Downloading GLORYS12V1 temperature: {start_date} to {end_date}")
    print(f"Region: {LAT_RANGE[0]}N-{LAT_RANGE[1]}N, {LON_RANGE[0]}E-{LON_RANGE[1]}E")

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["thetao"],  # Sea water potential temperature
        minimum_longitude=LON_RANGE[0],
        maximum_longitude=LON_RANGE[1],
        minimum_latitude=LAT_RANGE[0],
        maximum_latitude=LAT_RANGE[1],
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        minimum_depth=0.0,
        maximum_depth=1100.0,
        output_filename=output_file,
        output_directory=output_dir,
        username=username or os.environ.get("COPERNICUSMARINE_USER"),
        password=password or os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    print(f"Downloaded GLORYS data to: {output_file}")
    return output_file


def regrid_glorys_to_standard(
    glorys_path: str,
    target_depths: np.ndarray = STANDARD_DEPTHS,
    target_resolution: float = TARGET_RESOLUTION,
) -> xr.Dataset:
    """
    Regrids GLORYS12V1 temperature from native 1/12 deg to the standard 0.25 deg grid
    and interpolates to the 15 standard oceanographic depth levels.

    Args:
        glorys_path: Path to downloaded GLORYS NetCDF file
        target_depths: Standard depth levels to interpolate onto
        target_resolution: Target spatial resolution in degrees

    Returns:
        xarray Dataset with temperature on the standardized grid
    """
    ds = xr.open_dataset(glorys_path)

    # Define target grid coordinates
    target_lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-5, target_resolution)
    target_lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-5, target_resolution)

    # Spatial regridding via bilinear interpolation
    ds_regridded = ds.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
    )

    # Vertical interpolation to standard depth levels
    ds_standard = ds_regridded.interp(
        depth=target_depths,
        method="linear",
    )

    return ds_standard


def load_glorys_training_targets(
    glorys_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and prepares GLORYS12V1 data as training targets.

    Returns:
        lats: Array of latitude values
        lons: Array of longitude values
        temperature: Array of shape [Time, 15, Lat, Lon] in degrees Celsius
    """
    ds = regrid_glorys_to_standard(glorys_path)

    temperature = ds["thetao"].values  # [Time, Depth, Lat, Lon]
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    print(f"Loaded GLORYS training targets: {temperature.shape}")
    print(f"  Time steps: {temperature.shape[0]}")
    print(f"  Depth levels: {temperature.shape[1]}")
    print(f"  Spatial grid: {temperature.shape[2]} x {temperature.shape[3]}")

    return lats, lons, temperature.astype(np.float32)
