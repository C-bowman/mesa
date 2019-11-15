
from numpy import array, concatenate, mean, zeros
from pandas import read_hdf

from runpy import run_path
from os.path import isfile
from sys import argv

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior, reset_solps
from inference.gp_tools import GpOptimiser, GpRegressor
from inference.mcmc import GibbsChain, ParallelTempering
from inference.pdf_tools import BinaryTree

def bounds_transform(v, bounds, inverse=False):
    if inverse:
        return array([b[0] + (b[1]-b[0])*k for k,b in zip(v, bounds)])
    else:
        return array([(k-b[0])/(b[1]-b[0]) for k, b in zip(v, bounds)])

def grid_transform(point):
    tree = BinaryTree(limits = (0.,1.), layers = 6)
    return tuple([tree.lookup(v)[2] for v in point])

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
        'trust_region', 'trust_region_width', 'fixed_parameter_values', 'set_divertor_transport']

for key in keys:
    if key not in settings:
        raise ValueError('"{}" was not found in the settings module'.format(key))

# data & results filepaths
run_directory = settings['solps_run_directory']
ref_directory = settings['solps_ref_directory']
output_directory = settings['solps_output_directory']
training_data_file = settings['training_data_file']
diagnostic_data_file = settings['diagnostic_data_file']
diagnostic_data_desc = settings['diagnostic_data_desc']

# SOLPS settings
solps_n_species = settings['solps_n_species']
solps_n_timesteps = settings['solps_n_timesteps']
solps_dt = settings['solps_dt']
solps_n_proc = settings['solps_n_proc']
solps_iter_reset = settings['solps_iter_reset']
set_divertor_transport = settings['set_divertor_transport']

# optimiser settings
max_iterations = settings['max_iterations']
initial_sample_count = settings['initial_sample_count']
fixed_parameter_values = settings['fixed_parameter_values']
optimisation_bounds = settings['optimisation_bounds']
acquisition_function = settings['acquisition_function']
normalise_training_data = settings['normalise_training_data']
cross_validation = settings['cross_validation']
covariance_kernel = settings['covariance_kernel']
trust_region = settings['trust_region']
trust_region_width = settings['trust_region_width']




# build the indices for the varied vs fixed parameters:
varied_inds  = array([ i for i,v in enumerate(fixed_parameter_values) if v is None])
fixed_inds   = array([ i for i,v in enumerate(fixed_parameter_values) if v is not None])
fixed_values = array([ v for i,v in enumerate(fixed_parameter_values) if v is not None])



# start the optimisation loop
while True:
    # load the training data
    df = read_hdf(output_directory + training_data_file, 'training')
    # break the loop if we've hit the max number of iterations
    if df['iteration'].max() >= max_iterations: break
    # extract the training data
    log_posterior = df['log_posterior'].to_numpy().copy()

    parameters = []
    for X, D, H in zip(df['conductivity_parameters'], df['diffusivity_parameters'], df['div_parameters']):
        parameters.append( concatenate([X,D,H]) )

    # convert the data to the normalised coordinates:
    normalised_parameters = [bounds_transform(p,optimisation_bounds) for p in parameters]

    # build the set of grid-transformed points
    grid_set = set( grid_transform(p) for p in normalised_parameters )

    # get the training points by extracting the parameters which are to be optimised
    # from the normalised parameter vectors:
    training_points = [v[varied_inds] for v in normalised_parameters]

    # if requested, normalise the training data to have zero mean
    if normalise_training_data:
        data_mean = mean(log_posterior)
        log_posterior -= data_mean

    # construct the GP
    GP = GpRegressor(training_points, log_posterior, cross_val = cross_validation, kernel = covariance_kernel)

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
        max_point = training_points[max_ind]
        search_bounds = [(max(0., v-trhw), min(1., v+trhw)) for v in max_point]
    else:
        search_bounds = [(0.,1.) for i in range(len(varied_inds))]

    # build the GP-optimiser
    GPopt = GpOptimiser(training_points, log_posterior, hyperpars = mode, bounds=search_bounds,
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

    # convert the new point to a new parameter vector
    new_normalised_parameters = zeros(len(fixed_parameter_values))
    new_normalised_parameters[varied_inds] = new_point

    # back-transform to get the new point as model parameters
    new_parameters = bounds_transform(new_normalised_parameters, optimisation_bounds, inverse=True)

    # now insert the values of the fixed parameters
    new_parameters[fixed_inds] = fixed_values

    # check to see if the grid-transformed new point is already in the evaluated set
    if grid_transform(bounds_transform(new_parameters, optimisation_bounds)) in grid_set:
        raise ValueError(
            """
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            The latest proposed evaluation is a point which has been
            previously evaluated - this may indicate that a local
            maximum has been reached.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            """
        )

    # add the mean back to the prediction if the data was normalised
    if normalise_training_data:
        mu_lp += data_mean

    # produce transport profiles defined by new point
    radius = profile_radius_axis()

    chi = linear_transport_profile(radius, new_parameters[0:9])
    D = linear_transport_profile(radius, new_parameters[9:18])
    dna = new_parameters[18]
    hci = new_parameters[19]
    hce = new_parameters[20]

    # get the current iteration number
    i = df['iteration'].max()+1

    # Run SOLPS for the new point
    run_id = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, dna = dna, hci = hci, hce = hce,
                       run_directory = run_directory, output_directory = output_directory,
                       solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt,
                       n_proc = solps_n_proc, n_species = solps_n_species, set_div_transport = set_divertor_transport)

    # evaluate the chi-squared
    new_log_posterior = evaluate_log_posterior(iteration = i, directory = output_directory,
                                           diagnostic_data_file = diagnostic_data_file,
                                           diagnostic_data_desc = diagnostic_data_desc)

    # build a new row for the dataframe
    row_dict = {
        'iteration' : i,
        'conductivity_parameters' : new_parameters[0:9],
        'diffusivity_parameters' : new_parameters[9:18],
        'div_parameters' : new_parameters[18:21],
        'log_posterior' : new_log_posterior,
        'prediction_mean' : mu_lp,
        'prediction_error' : sigma_lp,
        'expected_fractional_improvement' : convergence
    }

    df.loc[df.index.max()+1] = row_dict  # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data
    del df

    if i % solps_iter_reset == 0:
        reset_solps(run_directory,ref_directory)
