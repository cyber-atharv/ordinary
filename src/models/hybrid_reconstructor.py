"""
HyperOcean-Mamba (HO-Mamba) Master Hybrid Reconstructor
Integrates:
1. Multi-Modal Surface State-Space Encoder (Ocean-Mamba)
2. Neural 4D-Var In-Situ Prompting Module (Argo Assimilator)
3. Analytical Sturm-Liouville Baroclinic Synthesizer (0-1000m across 15 depths)
4. Multi-Quantile Uncertainty Heads (10%, 50%, 90% confidence bounds)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from src.models.ocean_mamba import OceanMambaEncoder
from src.models.in_situ_prompting import InSituPromptingBlock
from src.data.sturm_liouville import BaroclinicSynthesizer, STANDARD_DEPTHS


class HyperOceanMamba(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        latent_dim: int = 128,
        num_modes: int = 5,
        depths = STANDARD_DEPTHS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        
        # 1. Surface State-Space Encoder
        self.surface_encoder = OceanMambaEncoder(in_channels=in_channels, latent_dim=latent_dim)
        
        # 2. In-Situ Prompting Module
        self.in_situ_block = InSituPromptingBlock(latent_dim=latent_dim)
        
        # 3. Baroclinic Normal Mode Projector (predicts modal amplitude coefficients a_0..a_4)
        self.modal_head = nn.Sequential(
            nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_modes, kernel_size=1)
        )
        
        # 4. Multi-Quantile Uncertainty Head (predicts lower 10% and upper 90% spread)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(latent_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 15 * 2, kernel_size=1)  # 15 depths lower spread + 15 depths upper spread
        )
        
        # 5. Physics-Based Modal Synthesizer
        self.baroclinic_synthesizer = BaroclinicSynthesizer(num_modes=num_modes, depths=depths)

    def forward(
        self, 
        surface_tensor: torch.Tensor, 
        argo_prompts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            surface_tensor: [B, 7, H, W] (SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)
            argo_prompts: Optional [B, Num_Floats, 17] live Argo float prompt tokens
        Returns:
            Dictionary containing:
                t_pred_50: [B, 15, H, W] median reconstructed 3D temperature field
                t_pred_10: [B, 15, H, W] lower 10% uncertainty bound
                t_pred_90: [B, 15, H, W] upper 90% uncertainty bound
                modal_amplitudes: [B, 5, H, W] predicted baroclinic mode coefficients
        """
        # Step 1: Encode multi-scale surface observations via Mamba State-Space blocks
        latent = self.surface_encoder(surface_tensor)
        
        # Step 2: In-Situ Neural 4D-Var float assimilation
        assimilated_latent = self.in_situ_block(latent, argo_prompts)
        
        # Step 3: Project to Baroclinic Normal Mode dynamic amplitudes
        modal_amps = self.modal_head(assimilated_latent)  # [B, 5, H, W]
        
        # Step 4: Reconstruct 3D Continuous Temperature Field using Sturm-Liouville eigenmodes
        t_median = self.baroclinic_synthesizer(modal_amps)  # [B, 15, H, W]
        
        # Step 5: Compute Quantile Confidence Bounds
        spread = torch.exp(self.uncertainty_head(assimilated_latent))  # Positive spread
        spread_lower = spread[:, :15, :, :]
        spread_upper = spread[:, 15:, :, :]
        
        t_lower = t_median - 0.2 * spread_lower
        t_upper = t_median + 0.2 * spread_upper
        
        return {
            "t_pred_50": t_median,
            "t_pred_10": t_lower,
            "t_pred_90": t_upper,
            "modal_amplitudes": modal_amps
        }
