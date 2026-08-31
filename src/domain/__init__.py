"""
INCOIS Operational & Ecosystem Domain Modules for OceanEmbed-X
"""
from src.domain.cyclone_tchp import compute_tchp_and_d26_numpy
from src.domain.pfz_upwelling import compute_d20_isotherm_depth, analyze_pfz_and_upwelling
from src.domain.oil_and_plastic import analyze_oil_spill_and_plastic_dispersion
from src.domain.active_sampling import recommend_optimal_float_drops
