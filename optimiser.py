
from numpy import array, concatenate, mean, zeros
from pandas import read_hdf
from sys import argv
import logging

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior, reset_solps
from input_parsing import parse_inputs, check_dependencies, logger_setup

from inference.gp_tools import GpOptimiser, GpRegressor
from inference.pdf_tools import BinaryTree

def bounds_transform(v, bounds, inverse=False):
    if inverse:
        return array([b[0] + (b[1]-b[0])*k for k,b in zip(v, bounds)])
    else:
        return array([(k-b[0])/(b[1]-b[0]) for k, b in zip(v, bounds)])

def grid_transform(point):
    tree = BinaryTree(limits = (0.,1.), layers = 6)
    return tuple([tree.lookup(v)[2] for v in point])

# check the validity of the input file and return its contents
settings = parse_inputs(argv)

# Check other data files are present
check_dependencies(settings)

# set-up the log file
logger_setup(argv)

# data & results filepaths
run_directory = settings['solps_run_directory']
ref_directory = settings['solps_ref_directory']
output_directory = settings['solps_output_directory']
training_data_file = settings['training_data_file']
diagnostic_data_files = settings['diagnostic_data_files']
diagnostic_data_observables = settings['diagnostic_data_observables']
diagnostic_data_errors = settings['diagnostic_data_errors']

# SOLPS settings
solps_n_species = settings['solps_n_species']
solps_n_timesteps = settings['solps_n_timesteps']
solps_dt = settings['solps_dt']
solps_n_proc = settings['solps_n_proc']
solps_iter_reset = settings['solps_iter_reset']
set_divertor_transport = settings['set_divertor_transport']
solps_timeout_hours = settings['solps_timeout_hours']

# optimiser settings
max_iterations = settings['max_iterations']
initial_sample_count = settings['initial_sample_count']
fixed_parameter_values = settings['fixed_parameter_values']
optimisation_bounds = settings['optimisation_bounds']
acquisition_function = settings['acquisition_function']
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
    if df['iteration'].max() >= max_iterations:
        logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
        break

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
    data_mean = mean(log_posterior[:initial_sample_count])
    log_posterior -= data_mean

    # construct the GP
    GP = GpRegressor(training_points, log_posterior, cross_val = cross_validation, kernel = covariance_kernel)
    bfgs_hps = GP.multistart_bfgs(starts=50)

    if GP.model_selector(bfgs_hps) > GP.model_selector(GP.hyperpars):
        mode = bfgs_hps
    else:
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
    GPopt = GpOptimiser(training_points, log_posterior, hyperpars = GP.hyperpars, bounds=search_bounds,
                        cross_val = cross_validation, kernel = covariance_kernel, acquisition=acquisition_function)

    # maximise the acquisition both by multi-start bfgs and differential evolution, and use the best of the two
    bfgs_prop = GPopt.propose_evaluation(bfgs=True)
    diff_prop = GPopt.propose_evaluation()

    bfgs_acq = GPopt.acquisition(bfgs_prop)
    diff_acq = GPopt.acquisition(diff_prop)

    if bfgs_acq > diff_acq:
        new_point = bfgs_prop
    else:
        new_point = diff_prop

    # calculate the convergence metric
    convergence = GPopt.acquisition.convergence_metric(new_point)

    # get predicted log-probability at the new point
    mu_lp, sigma_lp = GPopt.gp(new_point)

    # convert the new point to a new parameter vector
    new_normalised_parameters = zeros(len(fixed_parameter_values))
    new_normalised_parameters[varied_inds] = new_point

    # back-transform to get the new point as model parameters
    new_parameters = bounds_transform(new_normalised_parameters, optimisation_bounds, inverse=True)

    # now insert the values of the fixed parameters
    if len(fixed_inds) > 0: new_parameters[fixed_inds] = fixed_values

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

    # add the mean back to the prediction
    mu_lp += data_mean

    # produce transport profiles defined by new point
    radius = profile_radius_axis()

    chi = linear_transport_profile(radius, new_parameters[0:9])
    D = linear_transport_profile(radius, new_parameters[9:18])
    dna = new_parameters[18]
    hc = new_parameters[19]

    # get the current iteration number
    i = df['iteration'].max()+1

    # Run SOLPS for the new point
    run_status = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, dna = dna, hci = hc, hce = hc,
                           run_directory = run_directory, output_directory = output_directory,
                           solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt, timeout_hours = solps_timeout_hours,
                           n_proc = solps_n_proc, n_species = solps_n_species, set_div_transport = set_divertor_transport)

    if run_status == False:
        logging.info('[optimiser] Restoring SOLPS run directory from reference.')
        reset_solps(run_directory,ref_directory)
        logging.info('[optimiser] Restoration complete, trying new run...')
        continue

    # evaluate the chi-squared
    new_log_posterior = evaluate_log_posterior(iteration = i, directory = output_directory,
                                               diagnostic_data_files = diagnostic_data_files,
                                               diagnostic_data_observables = diagnostic_data_observables,
                                               diagnostic_data_errors = diagnostic_data_errors)

    # build a new row for the dataframe
    row_dict = {
        'iteration' : i,
        'conductivity_parameters' : new_parameters[0:9],
        'diffusivity_parameters' : new_parameters[9:18],
        'div_parameters' : new_parameters[18:20],
        'log_posterior' : new_log_posterior,
        'prediction_mean' : mu_lp,
        'prediction_error' : sigma_lp,
        'convergence_metric' : convergence
    }

    df.loc[df.index.max()+1] = row_dict  # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data
    del df

    if i % solps_iter_reset == 0:
        reset_solps(run_directory,ref_directory)
