"""
Pydantic API Schemas for OceanEmbed-X REST Server
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ProfilePredictionRequest(BaseModel):
    latitude: float = Field(..., ge=5.0, le=30.0, description="Target Latitude (5°N to 30°N)")
    longitude: float = Field(..., ge=45.0, le=105.0, description="Target Longitude (45°E to 105°E)")
    date: Optional[str] = Field("2023-08-15", description="Target Date (YYYY-MM-DD)")
    inject_live_floats: bool = Field(True, description="Enable In-Situ Neural 4D-Var float prompting")


class ProfilePredictionResponse(BaseModel):
    latitude: float
    longitude: float
    date: str
    depths: List[float]
    temperature_median: List[float]
    temperature_lower_10: List[float]
    temperature_upper_90: List[float]
    climatology_baseline: List[float]
    mld_m: float
    d26_m: float
    d20_m: float
    tchp_kj_cm2: float
    pfz_upwelling: Dict[str, Any]
    oil_and_plastic_risk: Dict[str, Any]


class DepthSliceRequest(BaseModel):
    depth_m: float = Field(200.0, description="Depth level in meters (0 to 1000m)")
    date: Optional[str] = Field("2023-08-15", description="Target Date (YYYY-MM-DD)")


class ActiveFloatModel(BaseModel):
    float_id: str
    latitude: float
    longitude: float
    qc_flag: int
    temperatures: List[float]
    depths: List[float]


class EmbeddingInspectionResponse(BaseModel):
    latitude: float
    longitude: float
    latent_dimension: int
    channel_statistics: List[Dict[str, Any]]
    dominant_mode_weights: List[float]
    spatial_eddy_energy: float
    summary: str
