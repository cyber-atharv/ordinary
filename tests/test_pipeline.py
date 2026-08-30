"""
Automated Unit and Pipeline Verification Tests for OceanEmbed-X
Tests:
1. Sturm-Liouville Baroclinic Mode Solver & Synthesizer
2. Mock Dataset Generator (7 surface channels, 15 depths)
3. HyperOcean-Mamba Forward Pass
4. Tropical Cyclone Heat Potential (TCHP & D26)
5. Mackenzie Tactical Sonar Sound Velocity Profiler
6. Depth-wise Metric Evaluation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.sturm_liouville import STANDARD_DEPTHS, solve_baroclinic_normal_modes, BaroclinicSynthesizer
from src.data.mock_generator import generate_north_indian_ocean_dataset
from src.models.hybrid_reconstructor import HyperOceanMamba
from src.domain.cyclone_tchp import compute_tchp_and_d26_numpy
from src.domain.tactical_sonar import mackenzie_sound_speed, analyze_tactical_acoustic_zones
from src.evaluation.metrics import compute_depthwise_metrics


def test_sturm_liouville_baroclinic_modes():
    modes, radii = solve_baroclinic_normal_modes(STANDARD_DEPTHS, num_modes=5)
    assert modes.shape == (5, 15), f"Expected shape (5, 15), got {modes.shape}"
    assert len(radii) == 5
    
    # Test Synthesizer
    synth = BaroclinicSynthesizer(num_modes=5, depths=STANDARD_DEPTHS)
    test_amps = torch.zeros(2, 5)  # zero amplitude should yield climatology
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


def test_cyclone_tchp():
    test_profile = np.array([29.0, 28.8, 28.5, 28.2, 28.0, 27.5, 26.5, 25.0, 22.0, 19.0, 15.0, 11.0, 8.0, 6.5, 5.5], dtype=np.float32)
    res = compute_tchp_and_d26_numpy(test_profile, STANDARD_DEPTHS)
    assert "tchp_kj_cm2" in res
    assert res["d26_m"] > 50.0  # 26 deg C isotherm around 75-100m
    assert res["tchp_kj_cm2"] > 0.0


def test_tactical_sonar():
    test_temp = np.array([28.0, 27.5, 27.0, 26.0, 24.0, 20.0, 16.0, 13.0, 11.0, 9.5, 8.0, 6.5, 5.5, 5.0, 4.8], dtype=np.float32)
    svp = mackenzie_sound_speed(test_temp)
    assert len(svp) == 15
    assert svp[0] > 1530.0  # warm surface sound speed ~1540 m/s
    
    zones = analyze_tactical_acoustic_zones(test_temp)
    assert "sofar_axis_depth_m" in zones
    assert "submarine_shadow_zone" in zones


def test_depthwise_metrics():
    t_true = np.ones((5, 15)) * 20.0
    t_pred = t_true + 0.1
    metrics = compute_depthwise_metrics(t_pred, t_true)
    assert len(metrics["rmse_per_depth_degC"]) == 15
    assert metrics["mean_overall_rmse"] < 0.2


if __name__ == "__main__":
    print("Running pipeline unit tests...")
    test_sturm_liouville_baroclinic_modes()
    print("  [PASS] Sturm-Liouville Baroclinic Mode Solver")
    test_mock_generator()
    print("  [PASS] Mock Dataset Generator")
    test_hyperocean_mamba_forward()
    print("  [PASS] HyperOcean-Mamba Forward Pass")
    test_cyclone_tchp()
    print("  [PASS] Tropical Cyclone Heat Potential (TCHP & D26)")
    test_tactical_sonar()
    print("  [PASS] Mackenzie Tactical Sonar Sound Velocity Profiler")
    test_depthwise_metrics()
    print("  [PASS] Depth-wise Metric Evaluation")
    print("All 6 test suites passed successfully!")
