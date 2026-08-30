"""Data ingestion, QC, and harmonization modules for OceanEmbed-X."""
from src.data.sturm_liouville import STANDARD_DEPTHS, solve_baroclinic_normal_modes, BaroclinicSynthesizer
from src.data.mock_generator import generate_north_indian_ocean_dataset, save_mock_dataset
from src.data.kaggle_loader import KaggleOceanDataLoader
