


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

# description of the error data stored in the data file - ene, ete, eti, eprad or ejsat
diagnostic_data_observables = [['ene', 'ete'],
                               ['ejsat']]





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
set_divertor_transport = False

# Number of species in SOLPS simulations
solps_n_species = 9



# ----------------------------------------------------------------------------
#   gaussian-process regression settings
# ----------------------------------------------------------------------------

# specifies whether the training data is normalised to have zero mean
# before modelling it using the Gaussian process
normalise_training_data = True

# Boolean flag to set whether cross-validation should be used in place
# of the marginal likelihood to select the GP hyper-parameters
cross_validation = False

# Import one of the covariance functions from gp_tools as the
# covariance_kernel variable so it can be passed to the GP.
from inference.gp_tools import SquaredExponential
covariance_kernel = SquaredExponential





# ----------------------------------------------------------------------------
#   optimiser settings
# ----------------------------------------------------------------------------

# Number of random-search evaluations which will be used to create an
# initial set of training data for the GP-optimisation
initial_sample_count = 25

# Maximum number of iterations after which the optimisation terminates
max_iterations = 200

# specifies what criteria is used to select new proposed evaluations
acquisition_function = 'max_prediction'

# Select whether or not a trust-region approach is used.
trust_region = True

# The width of the trust-region.
trust_region_width = 0.08

# Lower & upper bounds placed on the values of the
# profile model parameters
optimisation_bounds = [
    # Chi-profile parameter boundaries
    (0.,2.),   # left boundary height from barrier level
    (0.,2.),   # right boundary height from barrier level
    (0.,1.),   # left-middle height as a fraction of barrier-boundary gap
    (0.,1.),   # right-middle height as a fraction of barrier-boundary gap
    (-0.01, 0.01), # transport barrier centre
    (0.05, 0.4),   # transport barrier height
    (0.002, 0.04), # transport barrier width
    (1e-3, 0.05),  # radius gap between left-midpoint and transport barrier
    (1e-3, 0.05),  # radius gap between right-midpoint and transport barrier

    # D-profile parameter boundaries
    (0.,2.),   # left boundary height from barrier level
    (0.,2.),   # right boundary height from barrier level
    (0.,1.),   # left-middle height as a fraction of barrier-boundary gap
    (0.,1.),   # right-middle height as a fraction of barrier-boundary gap
    (-0.01, 0.01), # transport barrier centre
    (0.05, 0.4),   # transport barrier height
    (0.002, 0.04), # transport barrier width
    (1e-3, 0.05),  # radius gap between left-midpoint and transport barrier
    (1e-3, 0.05),  # radius gap between right-midpoint and transport barrier

    # Divertor transport boundaries
    (0.1,50.0),   # radial particle diffusion
    (0.1,50.0),   # radial ion heat diffusion
    (0.1,50.0),   # radial electron heat diffusion
]

# The 'fixed_parameter_values' list allows a sub-set of the parameters to be fixed at
# particular values, thereby removing them from the optimisation problem.

# If the value in the list is set to 'None', that parameter will be optimised as normal.
# If any value other than 'None' is given, the parameter will be fixed and the given value
# will be used in running SOLPS.

fixed_parameter_values = [
    # Chi-profile parameters
    None,   # left boundary height from barrier level
    None,   # right boundary height from barrier level
    None,   # left-middle height as a fraction of barrier-boundary gap
    None,   # right-middle height as a fraction of barrier-boundary gap
    None,   # transport barrier centre
    None,   # transport barrier height
    None,   # transport barrier width
    None,   # radius gap between left-midpoint and transport barrier
    None,   # radius gap between right-midpoint and transport barrier

    # D-profile parameters
    None,   # left boundary height from barrier level
    None,   # right boundary height from barrier level
    None,   # left-middle height as a fraction of barrier-boundary gap
    None,   # right-middle height as a fraction of barrier-boundary gap
    None,   # transport barrier centre
    None,   # transport barrier height
    None,   # transport barrier width
    None,   # radius gap between left-midpoint and transport barrier
    None,   # radius gap between right-midpoint and transport barrier

    # Divertor transport parameters
    None,   # radial particle diffusion
    None,   # radial ion heat diffusion
    None,   # radial electron heat diffusion
]
