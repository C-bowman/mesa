# ----------------------------------------------------------------------------
#   general settings
# ----------------------------------------------------------------------------

general_settings = {
    'ref_directory' : '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/', # directory where a reference SOLPS run is stored
    'training_data_file' : 'training_data.h5' # file name in which the training data will be stored
}


# ----------------------------------------------------------------------------
#   diagnostics settings
# ----------------------------------------------------------------------------
from numpy import load
from sims.instruments import ThomsonScattering
from mesa.diagnostics import WeightedObjective

instrument_data = load('# instrument data path #')
TS = ThomsonScattering(
    R=instrument_data['R'],
    z=instrument_data['z'],
    weights=instrument_data['weights'],
    measurements=load('# measurement data path #')
)

dgns = [TS]
wgts = [1.0]
objfn = WeightedObjective(dgns,wgts)

objective_function = [objfn]

# ----------------------------------------------------------------------------
#   simulation settings
# ----------------------------------------------------------------------------
from mesa.simulations import SOLPS
solps_settings = {
    'set_divertor_transport' : True,
    'transport_profile_bounds' : (-0.250, 0.240)
}
sim = SOLPS(
    exe='/pfs/work/g2hjame/solps-iter/software/solps'
    n_proc=6,
    timeout_hours=24, 
    concurrent_runs=10, 
    sim_settings=solps_setting
)

simulation = [sim]

# ----------------------------------------------------------------------------
#   driver settings
# ----------------------------------------------------------------------------

from mesa.driver import GPOptimizer
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
opt = 
    GPOptimizer(
        params,
        general_options=general_optimizer_options,
        specific_options=gpo_options
    )

driver = [opt]
