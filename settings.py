


# ----------------------------------------------------------------------------
#   directory settings
# ----------------------------------------------------------------------------

# directory where the SOLPS runs are stored
solps_run_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/gpfit/'

# directory where a reference SOLPS run is stored
solps_ref_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/'

# directory where the SOLPS output data and training data are stored
solps_output_directory = '/pfs/work/g2hjame/solpsopt_runs/tcv_58196_1.2s/'

# file name in which the training data will be stored
training_data_file = 'training_data.h5'

# file names in which the experimental data are stored
diagnostic_data_files = ['TCV_TS_58196_1200ms_combined', 'TCV_LP_58196_1200ms']

# description of the data stored in the data file - ne, te, ne_weighted_te, ti, prad or jsat
diagnostic_data_observables = [['ne', 'ne_weighted_te'],
                               ['jsat']]





# ----------------------------------------------------------------------------
#   SOLPS settings
# ----------------------------------------------------------------------------

# number of SOLPS time steps per iteration
solps_n_timesteps = 3000

# SOLPS time step
solps_dt = 5.0E-5

# Number of cores to execute SOLPS on
solps_n_proc = 6

# Number of optimiser iterations before cleaning the run directory
solps_iter_reset = 5

# Sets whether the divertor transport coefficients are overridden by those
# used for the PFR
set_divertor_transport = True

# Number of species in SOLPS simulations
solps_n_species = 9

# Number of hours to leave SOLPS running before timing out
solps_timeout_hours = 5





# ----------------------------------------------------------------------------
#   gaussian-process regression settings
# ----------------------------------------------------------------------------

# Boolean flag to set whether cross-validation should be used in place
# of the marginal likelihood to select the GP hyper-parameters
cross_validation = False

# Import one of the covariance functions from gp_tools as the
# covariance_kernel variable so it can be passed to the GP.
from inference.gp_tools import SquaredExponential
covariance_kernel = SquaredExponential

# Choose whether the errors on the experimental data are treated either as
# Gaussian or Cauchy:
error_model = 'gaussian'





# ----------------------------------------------------------------------------
#   optimiser settings
# ----------------------------------------------------------------------------

# Number of random-search evaluations which will be used to create an
# initial set of training data for the GP-optimisation
initial_sample_count = 25

# Maximum number of iterations after which the optimisation terminates
max_iterations = 200

# specifies what criteria is used to select new proposed evaluations
from inference.gp_tools import UpperConfidenceBound
acquisition_function = UpperConfidenceBound(kappa=1.)

# Select whether or not a trust-region approach is used.
trust_region = True

# The width of the trust-region.
trust_region_width = 0.08

# Lower & upper bounds placed on the values of the
# profile model parameters
optimisation_bounds = {
    # Chi-profile parameter boundaries
    'chi_boundary_left'  : (0.,2.),   # left boundary height from barrier level
    'chi_boundary_right' : (0.,2.),   # right boundary height from barrier level
    'chi_frac_left'      : (0.,1.),   # left-middle height as a fraction of barrier-boundary gap
    'chi_frac_right'     : (0.,1.),   # right-middle height as a fraction of barrier-boundary gap
    'chi_barrier_centre' : (-0.01, 0.01), # transport barrier centre
    'chi_barrier_height' : (0.05, 0.4),   # transport barrier height
    'chi_barrier_width'  : (0.002, 0.04), # transport barrier width
    'chi_gap_left'       : (1e-3, 0.05),  # radius gap between left-midpoint and transport barrier
    'chi_gap_right'      : (1e-3, 0.05),  # radius gap between right-midpoint and transport barrier

    # D-profile parameter boundaries
    'D_boundary_left'  : (0.,2.),   # left boundary height from barrier level
    'D_boundary_right' : (0.,2.),   # right boundary height from barrier level
    'D_frac_left'      : (0.,1.),   # left-middle height as a fraction of barrier-boundary gap
    'D_frac_right'     : (0.,1.),   # right-middle height as a fraction of barrier-boundary gap
    'D_barrier_centre' : (-0.01, 0.01), # transport barrier centre
    'D_barrier_height' : (0.05, 0.4),   # transport barrier height
    'D_barrier_width'  : (0.002, 0.04), # transport barrier width
    'D_gap_left'       : (1e-3, 0.05),  # radius gap between left-midpoint and transport barrier
    'D_gap_right'      : (1e-3, 0.05),  # radius gap between right-midpoint and transport barrier

    # Divertor transport boundaries
    'D_div'   : (0.1,50.0),   # radial particle diffusion
    'chi_div' : (0.1,50.0)    # radial heat diffusion
}

# The 'fixed_parameter_values' dictionary allows a sub-set of the parameters to be
# fixed at particular values, thereby removing them from the optimisation problem.

# If the value in the dictionary is set to 'None', that parameter will be optimised as normal.
# If any value other than 'None' is given, the parameter will be fixed and the given value
# will be used in running SOLPS.

fixed_parameter_values = {
    # Chi-profile parameter boundaries
    'chi_boundary_left'  : None,   # left boundary height from barrier level
    'chi_boundary_right' : None,   # right boundary height from barrier level
    'chi_frac_left'      : None,   # left-middle height as a fraction of barrier-boundary gap
    'chi_frac_right'     : None,   # right-middle height as a fraction of barrier-boundary gap
    'chi_barrier_centre' : None,   # transport barrier centre
    'chi_barrier_height' : None,   # transport barrier height
    'chi_barrier_width'  : None,   # transport barrier width
    'chi_gap_left'       : None,   # radius gap between left-midpoint and transport barrier
    'chi_gap_right'      : None,   # radius gap between right-midpoint and transport barrier

    # D-profile parameter boundaries
    'D_boundary_left'  : None,   # left boundary height from barrier level
    'D_boundary_right' : None,   # right boundary height from barrier level
    'D_frac_left'      : None,   # left-middle height as a fraction of barrier-boundary gap
    'D_frac_right'     : None,   # right-middle height as a fraction of barrier-boundary gap
    'D_barrier_centre' : None,   # transport barrier centre
    'D_barrier_height' : None,   # transport barrier height
    'D_barrier_width'  : None,   # transport barrier width
    'D_gap_left'       : None,   # radius gap between left-midpoint and transport barrier
    'D_gap_right'      : None,   # radius gap between right-midpoint and transport barrier

    # Divertor transport boundaries
    'D_div'   : None,   # radial particle diffusion
    'chi_div' : None    # radial heat diffusion
}
