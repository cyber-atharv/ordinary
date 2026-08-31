"""
Official INCOIS Data Ingestion, Harmonization, and Physics Modules for OceanEmbed-X
"""
from src.data.sturm_liouville import (
    STANDARD_DEPTHS,
    solve_baroclinic_normal_modes,
    BaroclinicSynthesizer,
    compute_standard_climatology_profile
)
from src.data.mock_generator import (
    generate_north_indian_ocean_dataset,
    save_mock_dataset
)
from src.data.glorys_loader import (
    download_glorys_temperature,
    load_glorys_training_targets,
    regrid_glorys_to_standard
)
from src.data.satellite_fetcher import (
    download_copernicus_variable,
    regrid_to_standard_grid,
    build_surface_tensor
)
from src.data.incois_argo_pipeline import (
    fetch_incois_gridded_argo,
    load_validation_profiles
)
