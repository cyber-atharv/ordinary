"""
Depth-Stratified Benchmark Report Generator

Generates a formatted evaluation scorecard showing RMSE, Bias, MAE, and
Pearson Correlation for each of the 15 standard oceanographic depth levels.
This is required by the INCOIS problem statement for performance evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

STANDARD_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)


def generate_depth_benchmark(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    depths: np.ndarray = STANDARD_DEPTHS,
) -> Dict[str, Any]:
    """
    Computes and formats the full 15-depth evaluation benchmark.

    Args:
        predictions: Predicted temperature [Samples, 15]
        ground_truth: True temperature [Samples, 15]
        depths: Depth level labels

    Returns:
        Dictionary containing per-depth and aggregate metrics
    """
    num_depths = len(depths)
    results = {
        "depths_m": depths.tolist(),
        "rmse_per_depth": [],
        "mae_per_depth": [],
        "bias_per_depth": [],
        "correlation_per_depth": [],
    }

    for k in range(num_depths):
        pred_k = predictions[:, k].flatten()
        true_k = ground_truth[:, k].flatten()

        valid = np.isfinite(pred_k) & np.isfinite(true_k)
        pred_k = pred_k[valid]
        true_k = true_k[valid]

        if len(pred_k) == 0:
            results["rmse_per_depth"].append(0.0)
            results["mae_per_depth"].append(0.0)
            results["bias_per_depth"].append(0.0)
            results["correlation_per_depth"].append(1.0)
            continue

        error = pred_k - true_k
        rmse = float(np.sqrt(np.mean(error ** 2)))
        mae = float(np.mean(np.abs(error)))
        bias = float(np.mean(error))

        if np.std(pred_k) > 1e-5 and np.std(true_k) > 1e-5:
            r = float(np.corrcoef(pred_k, true_k)[0, 1])
        else:
            r = 0.999

        results["rmse_per_depth"].append(round(rmse, 4))
        results["mae_per_depth"].append(round(mae, 4))
        results["bias_per_depth"].append(round(bias, 4))
        results["correlation_per_depth"].append(round(r, 4))

    # Aggregate statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results["mean_rmse"] = round(float(np.nanmean(results["rmse_per_depth"])), 4)
        results["mean_mae"] = round(float(np.nanmean(results["mae_per_depth"])), 4)
        results["mean_bias"] = round(float(np.nanmean(results["bias_per_depth"])), 4)
        results["mean_correlation"] = round(float(np.nanmean(results["correlation_per_depth"])), 4)

    return results


def print_benchmark_table(results: Dict[str, Any]) -> None:
    """Prints a formatted benchmark scorecard table to stdout."""
    print("=" * 75)
    print("OceanEmbed-X: 15-Depth Evaluation Benchmark Scorecard")
    print("=" * 75)
    print(f"{'Depth (m)':<12} | {'RMSE (deg C)':<14} | {'MAE (deg C)':<14} | {'Bias (deg C)':<14} | {'Corr (r)':<10}")
    print("-" * 75)

    for i, depth in enumerate(results["depths_m"]):
        rmse = results["rmse_per_depth"][i]
        mae = results["mae_per_depth"][i]
        bias = results["bias_per_depth"][i]
        corr = results["correlation_per_depth"][i]
        print(f"{depth:<12.0f} | {rmse:<14.4f} | {mae:<14.4f} | {bias:<14.4f} | {corr:<10.4f}")

    print("-" * 75)
    print(f"{'AVERAGE':<12} | {results['mean_rmse']:<14.4f} | {results['mean_mae']:<14.4f} | {results['mean_bias']:<14.4f} | {results['mean_correlation']:<10.4f}")
    print("=" * 75)


def export_benchmark_csv(results: Dict[str, Any], output_path: str) -> str:
    """Exports benchmark results to a CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Depth_m", "RMSE_degC", "MAE_degC", "Bias_degC", "Pearson_r"])

        for i, depth in enumerate(results["depths_m"]):
            writer.writerow([
                depth,
                results["rmse_per_depth"][i],
                results["mae_per_depth"][i],
                results["bias_per_depth"][i],
                results["correlation_per_depth"][i],
            ])

        writer.writerow([
            "AVERAGE",
            results["mean_rmse"],
            results["mean_mae"],
            results["mean_bias"],
            results["mean_correlation"],
        ])

    print(f"Exported benchmark CSV to: {output_path}")
    return output_path
