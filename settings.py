

# ----------------------------------------------------------------------------
#   directory settings
# ----------------------------------------------------------------------------

# directory where a reference SOLPS run is stored
solps_ref_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/'

# file name in which the training data will be stored
training_data_file = 'training_data.h5'


# ----------------------------------------------------------------------------
#   diagnostics settings
# ----------------------------------------------------------------------------
from numpy import load
from sims.instruments import ThomsonScattering

instrument_data = load('# instrument data path #')
TS = ThomsonScattering(
    R=instrument_data['R'],
    z=instrument_data['z'],
    weights=instrument_data['weights'],
    measurements=load('# measurement data path #')
)

diagnostics = [TS]


# ----------------------------------------------------------------------------
#   SOLPS settings
# ----------------------------------------------------------------------------

# Number of cores to execute SOLPS on
solps_n_proc = 6

# Sets whether the divertor transport coefficients are overridden by those
# used for the PFR
set_divertor_transport = True

# The range over which the transport coefficient profiles are defined
transport_profile_bounds = (-0.250, 0.240)

# the number of SOLPS runs which will be launched in parallel during initial sampling
concurrent_runs = 10

# Number of hours after which a SOLPS run will be automatically cancelled
solps_timeout_hours = 24


# ----------------------------------------------------------------------------
#   gaussian-process regression settings
# ----------------------------------------------------------------------------

# Boolean flag to set whether cross-validation should be used in place
# of the marginal likelihood to select the GP hyper-parameters
cross_validation = False

# Import one of the covariance functions from gp_tools as the
# covariance_kernel variable so it can be passed to the GP.
from inference.gp import SquaredExponential
covariance_kernel = SquaredExponential

from inference.gp import QuadraticMean
mean_function = QuadraticMean

# Choose whether the errors on the experimental data are treated either as
# Gaussian, Cauchy or Laplace:
error_model = 'cauchy'





# ----------------------------------------------------------------------------
#   optimiser settings
# ----------------------------------------------------------------------------

# Number of random-search evaluations which will be used to create an
# initial set of training data for the GP-optimisation
initial_sample_count = 30

# Maximum number of iterations after which the optimisation terminates
max_iterations = 200

# specifies which metric is used to select new proposed evaluations
from inference.gp import UpperConfidenceBound
acquisition_function = UpperConfidenceBound(kappa=1.)

# The width of the trust-region, set to None to disable the trust-region
trust_region_width = 0.3

# Lower & upper bounds placed on the values of the
# profile model parameters
optimisation_bounds = {
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

# The 'fixed_parameter_values' dictionary allows a sub-set of the parameters to be
# fixed at particular values, thereby removing them from the optimisation problem.

# If the value in the dictionary is set to 'None', that parameter will be optimised as normal.
# If any value other than 'None' is given, the parameter will be fixed and the given value
# will be used in running SOLPS.

fixed_parameter_values = {
    # Chi-profile parameters
    'chi_boundary_left'  : None,   # left boundary height from barrier level
    'chi_boundary_right' : None,   # right boundary height from barrier level
    'chi_frac_left'      : None,   # left-middle height as a fraction of barrier-boundary gap
    'chi_frac_right'     : None,   # right-middle height as a fraction of barrier-boundary gap
    'chi_barrier_centre' : None,   # transport barrier centre
    'chi_barrier_height' : None,   # transport barrier height
    'chi_barrier_width'  : None,   # transport barrier width
    'chi_gap_left'       : None,   # radius gap between left-midpoint and transport barrier
    'chi_gap_right'      : None,   # radius gap between right-midpoint and transport barrier

    # D-profile parameters
    'D_boundary_left'  : None,   # left boundary height from barrier level
    'D_boundary_right' : None,   # right boundary height from barrier level
    'D_frac_left'      : None,   # left-middle height as a fraction of barrier-boundary gap
    'D_frac_right'     : None,   # right-middle height as a fraction of barrier-boundary gap
    'D_barrier_centre' : None,   # transport barrier centre
    'D_barrier_height' : None,   # transport barrier height
    'D_barrier_width'  : None,   # transport barrier width
    'D_gap_left'       : None,   # radius gap between left-midpoint and transport barrier
    'D_gap_right'      : None,   # radius gap between right-midpoint and transport barrier
}
