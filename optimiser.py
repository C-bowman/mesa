
from numpy import array, concatenate, mean
from pandas import read_hdf

from runpy import run_path
from os.path import isfile
from sys import argv

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior
from inference.gp_tools import GpOptimiser, GpRegressor
from inference.mcmc import GibbsChain, ParallelTempering

def bounds_transform(v, bounds, inverse=False):
    if inverse:
        return array([b[0] + (b[1]-b[0])*k for k,b in zip(v, bounds)])
    else:
        return array([(k-b[0])/(b[1] - b[0]) for k, b in zip(v, bounds)])



# Get data from the settings module
if len(argv) == 1: # check to see if the settings module path was given
    raise ValueError('Path to settings module was not given as an argument')

if isfile(argv[1]): # check to see if the given path is valid
    settings = run_path(argv[1]) # run the settings module
else:
    raise ValueError('{} is not a valid path to a settings module'.format(argv[1]))

# check that the settings module contains all the required information
keys = ['solps_run_directory', 'solps_output_directory', 'optimisation_bounds', 'training_data_file',
        'diagnostic_data_file', 'diagnostic_data_desc', 'initial_sample_count','solps_n_timesteps','solps_dt',
        'acquisition_function', 'normalise_training_data', 'cross_validation', 'covariance_kernel',
        'trust_region', 'trust_region_width']

for key in keys:
    if key not in settings:
        raise ValueError('"{}" was not found in the settings module'.format(key))

# extract the required information from settings
run_directory = settings['solps_run_directory']
output_directory = settings['solps_output_directory']
optimisation_bounds = settings['optimisation_bounds']
training_data_file = settings['training_data_file']
diagnostic_data_file = settings['diagnostic_data_file']
diagnostic_data_desc = settings['diagnostic_data_desc']
initial_sample_count = settings['initial_sample_count']
solps_n_timesteps = settings['solps_n_timesteps']
solps_dt = settings['solps_dt']
solps_n_proc = settings['solps_n_proc']
max_iterations = settings['max_iterations']
acquisition_function = settings['acquisition_function']
normalise_training_data = settings['normalise_training_data']
cross_validation = settings['cross_validation']
covariance_kernel = settings['covariance_kernel']
trust_region = settings['trust_region']
trust_region_width = settings['trust_region_width']


# start the optimisation loop
while True:
    # load the training data
    df = read_hdf(output_directory + training_data_file, 'training')
    # break the loop if we've hit the max number of iterations
    if df['iteration'].max() >= max_iterations: break
    # extract the training data
    log_posterior = df['log_posterior'].to_numpy().copy()
    points = []
    for X, D in zip(df['conductivity_parameters'], df['diffusivity_parameters']):
        points.append( concatenate([X,D]) )

    # convert the data to the normalised coordinates:
    points = [bounds_transform(p,optimisation_bounds) for p in points]

    # if requested, normalise the training data to have zero mean
    if normalise_training_data:
        data_mean = mean(log_posterior)
        log_posterior -= data_mean

    # construct the GP
    GP = GpRegressor(points, log_posterior, cross_val = cross_validation, kernel = covariance_kernel)

    # define a set of temperature levels
    N_levels = 4
    temps = [10**(2.5*k/(N_levels-1.)) for k in range(N_levels)]

    # create a set of chains - one with each temperature
    chains = [ GibbsChain(posterior=GP.model_selector, start = GP.hyperpars, temperature=T) for T in temps ]
    hyperpar_bounds = GP.cov.get_bounds()
    for chain in chains:
        for i,b in enumerate(hyperpar_bounds): chain.set_boundaries(i, b)

    # When an instance of ParallelTempering is created, a dedicated process for each chain is spawned.
    # These separate processes will automatically make use of the available cpu cores, such that the
    # computations to advance the separate chains are performed in parallel.
    PT = ParallelTempering(chains=chains)

    PT.run_for(minutes=2)

    chains = PT.return_chains() # have all processes return their chain objects
    PT.shutdown() # shutdown the processes

    # extract the hyper-parameter estimate
    mode = chains[0].mode()

    # if MCMC found a better solution then use it
    de_score = GP.model_selector(GP.hyperpars)
    mc_score = GP.model_selector(mode)

    if mc_score < de_score:
        mode = GP.hyperpars

    # If a trust-region approach is being used, limit the search area
    # to a region around the current maximum
    trhw = 0.5*trust_region_width
    if trust_region:
        max_ind = log_posterior.argmax()
        max_point = points[max_ind]
        search_bounds = [(max(0., v-trhw), min(1., v+trhw)) for v in max_point]
    else:
        search_bounds = [(0.,1.) for i in range(18)]

    # build the GP-optimiser
    GPopt = GpOptimiser(points, log_posterior, hyperpars = mode, bounds=search_bounds,
                        cross_val = cross_validation, kernel = covariance_kernel)

    # get the new evaluation point by maximising the acquisition function
    if acquisition_function is 'expected_improvement':
        new_point, max_EI = GPopt.maximise_acquisition(GPopt.expected_improvement)
        convergence = abs(max_EI / GPopt.mu_max)
    elif acquisition_function is 'max_prediction':
        new_point, max_pred = GPopt.maximise_acquisition(GPopt.max_prediction)
        convergence = -1. - max_pred/GPopt.mu_max

    # get predicted chi-squared at the new point
    mu_lp, sigma_lp = GPopt.gp(new_point)

    # add the mean back to the prediction if the data was normalised
    if normalise_training_data:
        mu_lp += data_mean

    # back-transform to get the new point as model parameters
    new_point = bounds_transform(new_point, optimisation_bounds, inverse=True)

    # produce transport profiles defined by new point
    radius = profile_radius_axis()
    L = len(new_point)//2
    chi = linear_transport_profile(radius, new_point[:L])
    D = linear_transport_profile(radius, new_point[L:])

    # get the current iteration number
    i = df['iteration'].max()+1

    # Run SOLPS for the new point
    run_id = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, run_directory = run_directory,
                       output_directory = output_directory, solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt,
                       n_proc = solps_n_proc)

    # evaluate the chi-squared
    new_log_posterior = evaluate_log_posterior(iteration = i, directory = output_directory,
                                           diagnostic_data_file = diagnostic_data_file,
                                           diagnostic_data_desc = diagnostic_data_desc)

    # build a new row for the dataframe
    row_dict = {
        'iteration' : i,
        'conductivity_parameters' : new_point[:L],
        'diffusivity_parameters' : new_point[L:],
        'log_posterior' : new_log_posterior,
        'prediction_mean' : mu_lp,
        'prediction_error' : sigma_lp,
        'expected_fractional_improvement' : convergence
    }

    df.loc[df.index.max()+1] = row_dict  # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data
    del df