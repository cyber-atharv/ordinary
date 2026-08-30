"""
Kaggle Dataset Ingestion Engine for OceanEmbed-X
Downloads and loads supplementary Kaggle datasets using kagglehub:
1. NASA Ocean Climate (NetCDF surface fields)
2. Sentinel-2 Water Bodies (RGB masks)
3. Shifting Seas (Marine ecology / coral reef metrics)
"""

import os
from typing import Dict, Any, Optional

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False


class KaggleOceanDataLoader:
    """Manages downloading and locating supplementary Kaggle oceanographic datasets."""
    
    DATASET_REPOS = {
        "nasa_climate": "brsdincer/ocean-data-climate-change-nasa",
        "water_bodies": "franciscoescobar/satellite-images-of-water-bodies",
        "shifting_seas": "atharvasoundankar/shifting-seas-ocean-climate-and-marine-life-dataset"
    }

    def __init__(self, download_dir: Optional[str] = None):
        self.download_dir = download_dir
        self.cached_paths: Dict[str, str] = {}

    def download_dataset(self, key: str) -> Optional[str]:
        """Downloads a specific Kaggle dataset via kagglehub."""
        if not KAGGLEHUB_AVAILABLE:
            print("[KaggleLoader] 'kagglehub' package is not installed. Run 'pip install kagglehub'.")
            return None

        if key not in self.DATASET_REPOS:
            raise ValueError(f"Unknown dataset key: {key}. Available: {list(self.DATASET_REPOS.keys())}")

        repo = self.DATASET_REPOS[key]
        print(f"[KaggleLoader] Downloading latest version of '{repo}'...")
        try:
            path = kagglehub.dataset_download(repo)
            self.cached_paths[key] = path
            print(f"[KaggleLoader] Downloaded '{key}' to: {path}")
            return path
        except Exception as e:
            print(f"[KaggleLoader] Error downloading '{repo}': {e}")
            return None

    def download_all(self) -> Dict[str, str]:
        """Downloads all supplementary Kaggle datasets."""
        for key in self.DATASET_REPOS:
            self.download_dataset(key)
        return self.cached_paths


if __name__ == "__main__":
    loader = KaggleOceanDataLoader()
    print("[KaggleLoader] Initialized loader. Call loader.download_all() with valid Kaggle credentials.")
