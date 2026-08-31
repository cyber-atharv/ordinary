"""
Automated Unit and Pipeline Verification Tests for OceanEmbed-X (SIH26066)
Indian National Centre for Ocean Information Services (INCOIS) / MoES

Tests:
1. Sturm-Liouville Baroclinic Normal Mode Solver & Synthesizer
2. Multi-Source Satellite & In-Situ Mock Generator (7 surface channels, 15 depths)
3. HyperOcean-Mamba Forward Pass & Multi-Quantile Uncertainty Heads
4. Tropical Cyclone Heat Potential (TCHP & D26)
5. Potential Fishing Zone (PFZ) & D20 Thermocline Upwelling
6. INCOIS OOSA Oil Spill & Marine Plastic Submergence Analyzer
7. Intelligent Active Sampling Float Drop Optimizer
8. Satellite Latent Embedding Exporter & Stats
9. 15-Depth Verification Benchmark Scorecard Generator
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.sturm_liouville import STANDARD_DEPTHS, solve_baroclinic_normal_modes, BaroclinicSynthesizer
from src.data.mock_generator import generate_north_indian_ocean_dataset
from src.models.hybrid_reconstructor import HyperOceanMamba
from src.models.latent_embedder import LatentEmbeddingExporter
from src.domain.cyclone_tchp import compute_tchp_and_d26_numpy
from src.domain.pfz_upwelling import compute_d20_isotherm_depth, analyze_pfz_and_upwelling
from src.domain.oil_and_plastic import analyze_oil_spill_and_plastic_dispersion
from src.domain.active_sampling import recommend_optimal_float_drops
from src.evaluation.metrics import compute_depthwise_metrics
from src.evaluation.benchmark_report import generate_depth_benchmark, print_benchmark_table


def test_sturm_liouville_baroclinic_modes():
    modes, radii = solve_baroclinic_normal_modes(STANDARD_DEPTHS, num_modes=5)
    assert modes.shape == (5, 15), f"Expected shape (5, 15), got {modes.shape}"
    assert len(radii) == 5
    
    synth = BaroclinicSynthesizer(num_modes=5, depths=STANDARD_DEPTHS)
    test_amps = torch.zeros(2, 5)  # zero amplitude yields climatology baseline
    t_recon = synth(test_amps)
    assert t_recon.shape == (2, 15)
    assert t_recon[0, 0] > 27.0  # surface warm
    assert t_recon[0, -1] < 10.0  # deep cold


def test_mock_generator():
    data = generate_north_indian_ocean_dataset(num_days=2, resolution=0.5)
    assert data["surface_features"].shape[1] == 7
    assert data["ground_truth_3d"].shape[1] == 15
    assert len(data["argo_floats"]) > 0


def test_hyperocean_mamba_forward():
    model = HyperOceanMamba(in_channels=7, latent_dim=64, num_modes=5)
    model.eval()
    
    dummy_surf = torch.randn(2, 7, 16, 16)
    with torch.no_grad():
        out = model(dummy_surf)
        
    assert "t_pred_50" in out
    assert out["t_pred_50"].shape == (2, 15, 16, 16)
    assert out["t_pred_10"].shape == (2, 15, 16, 16)
    assert out["t_pred_90"].shape == (2, 15, 16, 16)
    assert out["modal_amplitudes"].shape == (2, 5, 16, 16)


def test_cyclone_tchp():
    test_profile = np.array([29.0, 28.8, 28.5, 28.2, 28.0, 27.5, 26.5, 25.0, 22.0, 19.0, 15.0, 11.0, 8.0, 6.5, 5.5], dtype=np.float32)
    res = compute_tchp_and_d26_numpy(test_profile, STANDARD_DEPTHS)
    assert "tchp_kj_cm2" in res
    assert res["d26_m"] > 50.0
    assert res["tchp_kj_cm2"] > 0.0


def test_pfz_and_d20_upwelling():
    test_profile = np.array([29.0, 28.5, 27.5, 25.0, 22.0, 19.0, 16.0, 13.0, 11.0, 9.5, 8.0, 6.5, 5.5, 5.0, 4.8], dtype=np.float32)
    d20 = compute_d20_isotherm_depth(test_profile, STANDARD_DEPTHS)
    assert 20.0 <= d20 <= 80.0
    
    pfz = analyze_pfz_and_upwelling(test_profile, 15.0, 70.0, STANDARD_DEPTHS)
    assert "d20_isotherm_depth_m" in pfz
    assert "pfz_suitability_score" in pfz
    assert pfz["pfz_suitability_score"] > 0


def test_oil_spill_and_plastic_dispersion():
    test_profile = np.array([28.5, 28.4, 28.2, 27.8, 27.5, 26.0, 22.0, 18.0, 14.0, 11.0, 8.5, 7.0, 6.0, 5.5, 5.0], dtype=np.float32)
    res = analyze_oil_spill_and_plastic_dispersion(test_profile, 15.0, 70.0, wind_speed_ms=7.5, depths=STANDARD_DEPTHS)
    assert "oil_spill_oosa" in res
    assert "plastic_debris_analysis" in res
    assert res["oil_spill_oosa"]["max_droplet_entrainment_depth_m"] > 0
    assert "Euphotic" in res["plastic_debris_analysis"]["dominant_vertical_zone"] or "Surface" in res["plastic_debris_analysis"]["dominant_vertical_zone"]


def test_active_sampling_recommendations():
    lats = np.array([10.0, 15.0, 20.0])
    lons = np.array([65.0, 70.0, 75.0])
    unc_vol = np.random.uniform(0.5, 2.5, (15, 3, 3))
    is_land = np.zeros((3, 3), dtype=bool)
    
    recs = recommend_optimal_float_drops(lats, lons, unc_vol, is_land, top_k=2)
    assert len(recs) == 2
    assert recs[0]["rank"] == 1
    assert "latitude" in recs[0]


def test_latent_embedding_exporter():
    model = HyperOceanMamba(in_channels=7, latent_dim=32, num_modes=5)
    exporter = LatentEmbeddingExporter(model)
    dummy_surf = np.random.randn(1, 7, 16, 16).astype(np.float32)
    emb = exporter.extract_embeddings(dummy_surf)
    assert emb.shape == (1, 32, 16, 16)
    
    stats = exporter.compute_embedding_statistics(emb)
    assert stats["latent_dimension"] == 32
    assert "mean_activation" in stats


def test_benchmark_report():
    t_true = np.random.uniform(5, 30, (20, 15))
    t_pred = t_true + np.random.normal(0, 0.4, (20, 15))
    results = generate_depth_benchmark(t_pred, t_true)
    assert len(results["rmse_per_depth"]) == 15
    assert len(results["correlation_per_depth"]) == 15
    assert results["mean_rmse"] > 0


if __name__ == "__main__":
    print("Running OceanEmbed-X automated test suite...")
    test_sturm_liouville_baroclinic_modes()
    print("  [PASS] 1. Sturm-Liouville Baroclinic Mode Solver")
    test_mock_generator()
    print("  [PASS] 2. Multi-Source Satellite & In-Situ Mock Generator")
    test_hyperocean_mamba_forward()
    print("  [PASS] 3. HyperOcean-Mamba Forward Pass & Quantile Heads")
    test_cyclone_tchp()
    print("  [PASS] 4. Tropical Cyclone Heat Potential (TCHP & D26)")
    test_pfz_and_d20_upwelling()
    print("  [PASS] 5. Potential Fishing Zone (PFZ) & D20 Upwelling")
    test_oil_spill_and_plastic_dispersion()
    print("  [PASS] 6. INCOIS OOSA Oil Spill & Plastic Dispersion")
    test_active_sampling_recommendations()
    print("  [PASS] 7. Intelligent Active Sampling Float Recommender")
    test_latent_embedding_exporter()
    print("  [PASS] 8. Satellite Latent Embedding Exporter")
    test_benchmark_report()
    print("  [PASS] 9. 15-Depth Benchmark Scorecard Generator")
    print("All 9 test suites passed successfully!")
