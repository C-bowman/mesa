

# directory where the SOLPS runs are stored
solps_run_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/gpfit/'

# directory where a reference SOLPS run is stored
solps_ref_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/'

# directory where the SOLPS output data and training data are stored
solps_output_directory = '/pfs/work/g2hjame/solpsopt_runs/tcv_58196_1.2s/'

# file name in which the training data will be stored
training_data_file = 'training_data.h5'

# file name in which the experimental data are stored
diagnostic_data_file = ['TCV_TS_DATA_58196.mat','TCV_LP_DATA_58196','TCV_equil_58196.mat','1.2',
                        'TCV_equil_58196.mat','1.2','-0.005']

# description of the data stored in the data file
diagnostic_data_desc = ['Midplane TS','Divertor LP','Equilibrium','Data Time','Grid Equilibrium','Grid Time','TS Z shift']

# number of SOLPS time steps per iteration
solps_n_timesteps = 3000

# SOLPS time step
solps_dt = 5.0E-5

# Number of cores to execute SOLPS on
solps_n_proc = 6

# Number of optimiser iterations before clearning the run directory
solps_iter_reset = 5

# Allow the transport in the divertor to vary
fit_solps_div_transport = True

# Number of random-search evaluations which will be used to create an
# initial set of training data for the GP-optimisation
initial_sample_count = 15

# Maximum number of iterations after which the optimisation terminates
max_iterations = 200

# Number of threads to be used when searching for the maximum
# of the acquisition function
threads = 4

# specifies whether the training data is normalised to have zero mean
# before modelling it using the Gaussian process
normalise_training_data = True

# specifies what criteria is used to select new proposed evaluations
acquisition_function = 'max_prediction'

# Boolean flag to set whether cross-validation should be used in place
# of the marginal likelihood to select the GP hyper-parameters
cross_validation = False

# Import one of the covariance functions from gp_tools as the
# covariance_kernel variable so it can be passed to the GP.
from inference.gp_tools import SquaredExponential
covariance_kernel = SquaredExponential

# Select whether or not a trust-region approach is used.
trust_region = True

# The width of the trust-region.
trust_region_width = 0.08

# The total time (in seconds) which will be used to search
# for the maximum of the acquisition function
search_time = 600

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
