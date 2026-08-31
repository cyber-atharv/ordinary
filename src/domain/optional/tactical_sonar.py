"""
Naval Defense Tactical Sonar Intelligence Engine
Calculates:
1. Mackenzie Sound Velocity Profile (SVP): C(z) from T(z), S(z), and depth z
2. SOFAR Sound Channel Axis: Minimum sound velocity depth for long-range submarine detection
3. Surface Duct Thickness & Cutoff Frequency: Acoustic trapping for hull-mounted sonar
4. Hamiltonian Acoustic Ray Tracing: Projects submarine acoustic shadow zones
"""

import numpy as np
from typing import Dict, Any, List
from src.data.sturm_liouville import STANDARD_DEPTHS


def mackenzie_sound_speed(
    temperature: np.ndarray,
    salinity: np.ndarray = None,
    depths: np.ndarray = STANDARD_DEPTHS
) -> np.ndarray:
    """
    Computes acoustic speed of sound in seawater (m/s) using the 9-term Mackenzie (1981) formula.
    C(T, S, z) = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
                 + 1.340*(S - 35) + 1.630e-2*z + 1.675e-7*z^2
                 - 1.025e-2*T*(S - 35) - 7.139e-13*T*z^3
    """
    t = np.asarray(temperature, dtype=np.float32)
    z = np.asarray(depths, dtype=np.float32)
    s = np.full_like(t, 35.5) if salinity is None else np.asarray(salinity, dtype=np.float32)
    
    c = (
        1448.96 +
        4.591 * t -
        5.304e-2 * (t ** 2) +
        2.374e-4 * (t ** 3) +
        1.340 * (s - 35.0) +
        1.630e-2 * z +
        1.675e-7 * (z ** 2) -
        1.025e-2 * t * (s - 35.0) -
        7.139e-13 * t * (z ** 3)
    )
    return np.round(c, 2)


def analyze_tactical_acoustic_zones(
    temperature: np.ndarray,
    salinity: np.ndarray = None,
    depths: np.ndarray = STANDARD_DEPTHS
) -> Dict[str, Any]:
    """
    Analyzes naval acoustic ducting, shadow zones, and the SOFAR channel axis.
    """
    svp = mackenzie_sound_speed(temperature, salinity, depths)
    
    # 1. SOFAR Channel Axis (Depth of minimum sound speed)
    min_idx = int(np.argmin(svp))
    sofar_depth = float(depths[min_idx])
    sofar_speed = float(svp[min_idx])
    
    # 2. Surface Duct Thickness (Surface to layer where sound speed starts decreasing)
    surface_speed = float(svp[0])
    duct_thickness = 30.0
    for i in range(1, len(depths)):
        if svp[i] < svp[i - 1]:
            duct_thickness = float(depths[i - 1])
            break
            
    # 3. Submarine Acoustic Shadow Zone (Zone beneath surface duct before deep convergence)
    shadow_zone_top = duct_thickness
    shadow_zone_bottom = min(sofar_depth, 400.0)
    
    return {
        "sound_velocity_profile": svp.tolist(),
        "depths": depths.tolist(),
        "surface_sound_speed_ms": surface_speed,
        "sofar_axis_depth_m": sofar_depth,
        "sofar_minimum_speed_ms": sofar_speed,
        "surface_duct_thickness_m": duct_thickness,
        "submarine_shadow_zone": {
            "top_m": shadow_zone_top,
            "bottom_m": shadow_zone_bottom,
            "tactical_advantage": "Submarines cruising between 50m-200m remain shielded from surface active sonar due to downward ray refraction."
        }
    }


def compute_2d_acoustic_ray_paths(
    temperature: np.ndarray,
    source_depth: float = 15.0,  # Surface ship hull sonar depth (m)
    launch_angles_deg: List[float] = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0],
    range_max_km: float = 15.0,
    depths: np.ndarray = STANDARD_DEPTHS
) -> List[Dict[str, Any]]:
    """
    Computes acoustic ray traces using Snell's law in a stratified ocean.
    """
    svp = mackenzie_sound_speed(temperature, None, depths)
    c0 = float(np.interp(source_depth, depths, svp))
    
    rays = []
    for angle in launch_angles_deg:
        theta0 = np.radians(angle)
        snell_param = np.cos(theta0) / c0  # Snell's acoustic invariant
        
        # Ray integration
        r_points = [0.0]
        z_points = [source_depth]
        
        curr_r = 0.0
        curr_z = source_depth
        curr_theta = theta0
        ds = 50.0  # Step size (meters)
        
        while curr_r < range_max_km * 1000.0 and 0.0 <= curr_z <= 1000.0:
            c_curr = float(np.interp(curr_z, depths, svp))
            cos_theta = np.clip(snell_param * c_curr, -1.0, 1.0)
            curr_theta = np.arccos(cos_theta) if curr_theta >= 0 else -np.arccos(cos_theta)
            
            # Sound bends toward slower water (downward in thermocline)
            curr_r += ds * np.cos(curr_theta)
            curr_z += ds * np.sin(curr_theta)
            
            if curr_z < 0.0:
                curr_z = 0.0
                curr_theta = -curr_theta  # Surface reflection
            elif curr_z > 1000.0:
                curr_z = 1000.0
                curr_theta = -curr_theta  # Bottom reflection
                
            r_points.append(round(curr_r / 1000.0, 3))
            z_points.append(round(curr_z, 1))
            
        rays.append({
            "launch_angle_deg": angle,
            "range_km": r_points,
            "depth_m": z_points
        })
        
    return rays
