"""
Multi-Source Satellite Data Fetcher

Downloads and harmonizes the satellite surface observation products specified
by the INCOIS SIH26066 problem statement:
  - SST: OSTIA (0.05 deg daily) -> regrid to 0.25 deg
  - SSS: SMAP/SMOS (0.125 deg daily) -> regrid to 0.25 deg
  - SSH/SLA: DUACS (0.25 deg daily) -> already at target resolution
  - Currents: OSCAR L4 (0.25 deg daily) -> already at target resolution
  - Winds: CCMP v3.1 (0.25 deg 6-hourly) -> daily mean
"""

import os
import numpy as np
import xarray as xr
from typing import Optional


# Copernicus Marine product identifiers
PRODUCT_IDS = {
    "sst": {
        "dataset_id": "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        "variable": "analysed_sst",
        "doi": "https://doi.org/10.48670/moi-00168",
    },
    "sss": {
        "dataset_id": "MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013",
        "variable": "sos",
        "doi": "https://doi.org/10.48670/moi-00051",
    },
    "ssh": {
        "dataset_id": "SEALEVEL_GLO_PHY_L4_MY_008_047",
        "variable": "sla",
        "doi": "https://doi.org/10.48670/moi-00145",
    },
}

# North Indian Ocean bounding box
LAT_RANGE = (5.0, 30.0)
LON_RANGE = (45.0, 105.0)
TARGET_RESOLUTION = 0.25


def download_copernicus_variable(
    variable_key: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """
    Downloads a single surface variable from Copernicus Marine Service.

    Args:
        variable_key: One of 'sst', 'sss', 'ssh'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        output_dir: Output directory for NetCDF files
        username: Copernicus Marine credentials
        password: Copernicus Marine credentials

    Returns:
        Path to the downloaded file
    """
    try:
        import copernicusmarine
    except ImportError:
        raise ImportError("Install copernicusmarine: pip install copernicusmarine")

    if variable_key not in PRODUCT_IDS:
        raise ValueError(f"Unknown variable: {variable_key}. Options: {list(PRODUCT_IDS.keys())}")

    product = PRODUCT_IDS[variable_key]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{variable_key}_{start_date}_{end_date}.nc")

    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return output_file

    print(f"Downloading {variable_key.upper()} from Copernicus Marine...")

    copernicusmarine.subset(
        dataset_id=product["dataset_id"],
        variables=[product["variable"]],
        minimum_longitude=LON_RANGE[0],
        maximum_longitude=LON_RANGE[1],
        minimum_latitude=LAT_RANGE[0],
        maximum_latitude=LAT_RANGE[1],
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        output_filename=output_file,
        output_directory=output_dir,
        username=username or os.environ.get("COPERNICUSMARINE_USER"),
        password=password or os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    return output_file


def regrid_to_standard_grid(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """
    Regrids any satellite product to the standard 0.25 deg grid via bilinear interpolation.
    """
    target_lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-5, TARGET_RESOLUTION)
    target_lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-5, TARGET_RESOLUTION)

    # Handle different coordinate naming conventions across products
    lat_dim = None
    for candidate in ["latitude", "lat", "Latitude"]:
        if candidate in ds.dims or candidate in ds.coords:
            lat_dim = candidate
            break

    lon_dim = None
    for candidate in ["longitude", "lon", "Longitude"]:
        if candidate in ds.dims or candidate in ds.coords:
            lon_dim = candidate
            break

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot identify lat/lon coordinates in dataset: {list(ds.coords)}")

    return ds[var_name].interp({lat_dim: target_lats, lon_dim: target_lons}, method="linear")


def build_surface_tensor(
    sst_path: str,
    sss_path: str,
    ssh_path: str,
    currents_path: Optional[str] = None,
    winds_path: Optional[str] = None,
) -> np.ndarray:
    """
    Assembles the 7-channel surface input tensor from individual product files.

    Returns:
        Array of shape [Time, 7, Lat, Lon] with channels:
          0: SST (deg C)
          1: SSS (PSU)
          2: SSH/SLA (m)
          3: Zonal Current U (m/s)
          4: Meridional Current V (m/s)
          5: Zonal Wind U (m/s)
          6: Meridional Wind V (m/s)
    """
    print("Building 7-channel surface input tensor...")

    sst = regrid_to_standard_grid(xr.open_dataset(sst_path), "analysed_sst")
    sss = regrid_to_standard_grid(xr.open_dataset(sss_path), "sos")
    ssh = regrid_to_standard_grid(xr.open_dataset(ssh_path), "sla")

    num_time = len(sst.time)
    num_lat = len(sst.latitude) if "latitude" in sst.dims else len(sst.lat)
    num_lon = len(sst.longitude) if "longitude" in sst.dims else len(sst.lon)

    tensor = np.zeros((num_time, 7, num_lat, num_lon), dtype=np.float32)
    tensor[:, 0] = sst.values
    tensor[:, 1] = sss.values
    tensor[:, 2] = ssh.values

    if currents_path and os.path.exists(currents_path):
        ds_curr = xr.open_dataset(currents_path)
        tensor[:, 3] = regrid_to_standard_grid(ds_curr, "u").values
        tensor[:, 4] = regrid_to_standard_grid(ds_curr, "v").values

    if winds_path and os.path.exists(winds_path):
        ds_wind = xr.open_dataset(winds_path)
        tensor[:, 5] = regrid_to_standard_grid(ds_wind, "uwnd").values
        tensor[:, 6] = regrid_to_standard_grid(ds_wind, "vwnd").values

    print(f"Surface tensor shape: {tensor.shape} [Time, Channels, Lat, Lon]")
    return tensor
