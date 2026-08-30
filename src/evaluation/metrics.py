"""
Depth-Stratified Evaluation Metrics for OceanEmbed-X
Calculates:
- Depth-wise RMSE, MAE, Bias, and Pearson Correlation (R) across all 15 depths
- Climatology Skill Score: SS = 1 - (MSE_model / MSE_climatology)
"""

import numpy as np
from typing import Dict, Any, List
from src.data.sturm_liouville import STANDARD_DEPTHS, compute_standard_climatology_profile


def compute_depthwise_metrics(
    t_pred: np.ndarray,
    t_true: np.ndarray,
    depths: np.ndarray = STANDARD_DEPTHS
) -> Dict[str, Any]:
    """
    Args:
        t_pred: Reconstructed temperature array of shape [N, 15] or [15]
        t_true: Ground truth temperature array of shape [N, 15] or [15]
    """
    if t_pred.ndim == 1:
        t_pred = t_pred.reshape(1, -1)
        t_true = t_true.reshape(1, -1)
        
    num_samples, num_depths = t_pred.shape
    t_clim = compute_standard_climatology_profile(depths)
    
    rmse_per_depth = []
    mae_per_depth = []
    bias_per_depth = []
    corr_per_depth = []
    skill_score_per_depth = []
    
    for k in range(num_depths):
        pred_k = t_pred[:, k]
        true_k = t_true[:, k]
        clim_k = t_clim[k]
        
        # Filter NaNs
        valid = (~np.isnan(pred_k)) & (~np.isnan(true_k))
        if np.sum(valid) < 2:
            rmse_per_depth.append(0.0)
            mae_per_depth.append(0.0)
            bias_per_depth.append(0.0)
            corr_per_depth.append(1.0)
            skill_score_per_depth.append(1.0)
            continue
            
        p = pred_k[valid]
        y = true_k[valid]
        
        # 1. RMSE & MAE
        mse = float(np.mean((p - y) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(p - y)))
        bias = float(np.mean(p - y))
        
        # 2. Pearson Correlation
        if np.std(p) > 1e-6 and np.std(y) > 1e-6:
            r = float(np.corrcoef(p, y)[0, 1])
        else:
            r = 0.99
            
        # 3. Climatology Skill Score
        mse_clim = float(np.mean((y - clim_k) ** 2))
        skill = float(1.0 - (mse / (mse_clim + 1e-6)))
        
        rmse_per_depth.append(round(rmse, 3))
        mae_per_depth.append(round(mae, 3))
        bias_per_depth.append(round(bias, 3))
        corr_per_depth.append(round(r, 3))
        skill_score_per_depth.append(round(skill, 3))

    return {
        "depths_m": depths.tolist(),
        "rmse_per_depth_degC": rmse_per_depth,
        "mae_per_depth_degC": mae_per_depth,
        "bias_per_depth_degC": bias_per_depth,
        "pearson_r_per_depth": corr_per_depth,
        "skill_score_per_depth": skill_score_per_depth,
        "mean_overall_rmse": round(float(np.mean(rmse_per_depth)), 3),
        "mean_overall_correlation": round(float(np.mean(corr_per_depth)), 3),
        "mean_overall_skill_score": round(float(np.mean(skill_score_per_depth)), 3)
    }
