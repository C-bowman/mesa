# ----------------------------------------------------------------------------
#   general settings
# ----------------------------------------------------------------------------
# directory where a reference SOLPS run is stored
ref_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/'

# file name in which the training data will be stored
training_data_file = 'training_data.h5'

# ----------------------------------------------------------------------------
#   diagnostics settings
# ----------------------------------------------------------------------------
from numpy import load
from sims.instruments import ThomsonScattering
from mesa.simulations.solps import Solps, SolpsLikelihood

instrument_data = load('# instrument data path #')
TS = ThomsonScattering(
    R=instrument_data['R'],
    z=instrument_data['z'],
    weights=instrument_data['weights'],
    measurements=load('# measurement data path #')
)

objective_function = SolpsLikelihood(diagnostics=[TS])

# ----------------------------------------------------------------------------
#   simulation settings
# ----------------------------------------------------------------------------
simulation = Solps(
    exe='/pfs/work/g2hjame/solps-iter/software/solps',
    n_proc=6,
    timeout_hours=24,
    set_divertor_transport=True,
    transport_profile_bounds=(-0.250, 0.240)
)

# ----------------------------------------------------------------------------
#   strategy settings
# ----------------------------------------------------------------------------

from mesa.strategies import GPOptimizer
from inference.gp import SquaredExponential, QuadraticMean, UpperConfidenceBound

# parameters to fix or vary. To fix set as a single value. To vary set bounds as a tuple
params = {
    # Chi-profile parameter boundaries
    'chi_boundary_left'  : (0., 6.),       # left boundary height from barrier level
    'chi_boundary_right' : (0., 15.),      # right boundary height from barrier level
    'chi_frac_left'      : (0.25, 1.),     # left-middle height as a fraction of barrier-boundary gap
    'chi_frac_right'     : (0.25, 1.),     # right-middle height as a fraction of barrier-boundary gap
    'chi_barrier_centre' : (-0.05, 0.05),  # transport barrier centre
    'chi_barrier_height' : (1e-3, 0.2),    # transport barrier height
    'chi_barrier_width'  : (0.002, 0.04),  # transport barrier width
    'chi_gap_left'       : (2e-3, 0.05),   # radius gap between left-midpoint and transport barrier
    'chi_gap_right'      : (2e-3, 0.05),   # radius gap between right-midpoint and transport barrier

    # D-profile parameter boundaries
    'D_boundary_left'  : (0., 4.),       # left boundary height from barrier level
    'D_boundary_right' : (0., 2.),       # right boundary height from barrier level
    'D_frac_left'      : (0.25, 1.),     # left-middle height as a fraction of barrier-boundary gap
    'D_frac_right'     : (0.25, 1.),     # right-middle height as a fraction of barrier-boundary gap
    'D_barrier_centre' : (-0.05, 0.05),  # transport barrier centre
    'D_barrier_height' : (1e-3, 0.2),    # transport barrier height
    'D_barrier_width'  : (0.002, 0.04),  # transport barrier width
    'D_gap_left'       : (2e-3, 0.05),    # radius gap between left-midpoint and transport barrier
    'D_gap_right'      : (2e-3, 0.05),    # radius gap between right-midpoint and transport barrier
}
general_optimizer_options = {
    'initial_sample_count' : 30,
    'max_iterations' : 200
}
gpo_options = {
    'covariance_kernel' : SquaredExponential, 
    'mean_function' : QuadraticMean,
    'acquisition_function' : UpperConfidenceBound(kappa=1.),
    'cross_validation' : False,
    'error_model' : 'cauchy',
    'trust_region_width' : 0.3
}

strategy = GPOptimizer(
    params,
    initial_sample_count=0,
    max_iterations=200,
    concurrent_runs=10,
    covariance_kernel=SquaredExponential,
    mean_function=QuadraticMean,
    acquisition_function=UpperConfidenceBound(kappa=1.),
    cross_validation=False,
    trust_region_width=0.3
)
