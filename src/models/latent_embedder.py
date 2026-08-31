"""
Satellite Latent Embedding Exporter

Extracts, saves, and visualizes the intermediate latent embeddings produced by
the OceanMamba surface encoder. This module is required because the problem
statement specifically calls for a "Satellite Embedding-Based Framework" --
the embeddings must be inspectable and exportable.
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, Any


class LatentEmbeddingExporter:
    """
    Wraps the trained OceanMamba encoder to extract and export intermediate
    satellite embeddings from surface observation inputs.
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: Trained HyperOceanMamba model instance
            device: Compute device ('cpu' or 'cuda')
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, surface_tensor: np.ndarray) -> np.ndarray:
        """
        Extracts latent embeddings from surface satellite observations.

        Args:
            surface_tensor: Array of shape [Batch, 7, H, W] with surface channels:
                (SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)

        Returns:
            embeddings: Array of shape [Batch, LatentDim, H, W]
        """
        with torch.no_grad():
            x = torch.from_numpy(surface_tensor).float().to(self.device)
            latent = self.model.surface_encoder(x)
            return latent.cpu().numpy()

    def export_embeddings(
        self,
        surface_tensor: np.ndarray,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extracts and saves embeddings to a .npz file for analysis and visualization.

        Args:
            surface_tensor: Input surface observations [Batch, 7, H, W]
            output_path: Path to save the .npz file
            metadata: Optional metadata dictionary to include

        Returns:
            Path to the saved embedding file
        """
        embeddings = self.extract_embeddings(surface_tensor)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        save_dict = {
            "embeddings": embeddings,
            "embedding_shape": np.array(embeddings.shape),
        }

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, np.ndarray):
                    save_dict[key] = value
                else:
                    save_dict[key] = np.array(value)

        np.savez_compressed(output_path, **save_dict)
        print(f"Exported satellite embeddings to: {output_path}")
        print(f"  Shape: {embeddings.shape} (Batch, LatentDim, H, W)")
        print(f"  Embedding dimension: {embeddings.shape[1]}")

        return output_path

    def compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Computes summary statistics for the extracted embeddings.

        Args:
            embeddings: Array of shape [Batch, LatentDim, H, W]

        Returns:
            Dictionary of embedding statistics
        """
        return {
            "shape": embeddings.shape,
            "latent_dimension": int(embeddings.shape[1]),
            "mean_activation": float(np.mean(embeddings)),
            "std_activation": float(np.std(embeddings)),
            "min_activation": float(np.min(embeddings)),
            "max_activation": float(np.max(embeddings)),
            "sparsity": float(np.mean(np.abs(embeddings) < 0.01)),
            "channel_means": np.mean(embeddings, axis=(0, 2, 3)).tolist(),
        }
