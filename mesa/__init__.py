
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import array, log
from numpy.random import random
from pandas import DataFrame, read_hdf

from os.path import isfile
import logging

from mesa.parsing import parse_inputs, check_dependencies, logger_setup, check_error_model
from mesa.models import linear_transport_profile, profile_radius_axis
from mesa.solps import run_solps, evaluate_log_posterior, reset_solps
from mesa.parameters import conductivity_profile, diffusivity_profile

from inference.gp_tools import GpOptimiser, GpRegressor
from inference.pdf_tools import BinaryTree


def bounds_transform(v, bounds, inverse=False):
    if inverse:
        return array([b[0] + (b[1]-b[0])*k for k, b in zip(v, bounds)])
    else:
        return array([(k-b[0])/(b[1]-b[0]) for k, b in zip(v, bounds)])


def grid_transform(point):
    tree = BinaryTree(limits=(0., 1.), layers=6)
    return tuple([tree.lookup(v)[2] for v in point])


def hypercube_sample(bounds):
    return [b[0] + (b[1]-b[0])*random() for b in bounds]


def uniform_sample(bounds):
    return bounds[0] + (bounds[1]-bounds[0])*random()


def initial_sampling(settings_filepath):
    """
    Evaluates randomly selected points within the parameters
    space to create a set of initial training data for the GP.

    :param settings_filepath: The path to the settings file.
    :return:
    """
    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # Check other data files are present
    check_dependencies(settings, skip_training = True)

    # set-up the log file
    logger_setup(settings_filepath)

    # data & results filepaths
    run_directory = settings['solps_run_directory']
    output_directory = settings['solps_output_directory']
    ref_directory = settings['solps_ref_directory']
    training_data_file = settings['training_data_file']
    diagnostic_data_files = settings['diagnostic_data_files']
    diagnostic_data_observables = settings['diagnostic_data_observables']
    diagnostic_data_errors = settings['diagnostic_data_errors']

    # SOLPS settings
    solps_n_timesteps = settings['solps_n_timesteps']
    solps_dt = settings['solps_dt']
    solps_n_proc = settings['solps_n_proc']
    solps_iter_reset = settings['solps_iter_reset']
    solps_n_species = settings['solps_n_species']
    solps_timeout_hours = settings['solps_timeout_hours']
    set_divertor_transport = settings['set_divertor_transport']

    # optimiser settings
    fixed_parameter_values = settings['fixed_parameter_values']
    optimisation_bounds = settings['optimisation_bounds']
    initial_sample_count = settings['initial_sample_count']

    all_parameters = [key for key in fixed_parameter_values.keys()]
    free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]

    # first check if the training data file exists already, or needs to be created
    if not isfile(output_directory + training_data_file):
        # define what columns will be in the dataframe
        cols = [
            'iteration',
            'gaussian_logprob',
            'cauchy_logprob',
            'laplace_logprob',
            'logistic_logprob',
            'prediction_mean',
            'prediction_error',
            'convergence_metric',
            *all_parameters
        ]

        # create the empty dataframe to store the training data and save it to HDF
        df = DataFrame(columns=cols)
        df.to_hdf(output_directory + training_data_file, key='training', mode='w')
        del df

    # loop until enough samples have been evaluated
    while True:
        df = read_hdf(output_directory + training_data_file, 'training')

        # get the current iteration number
        i = 1 if df['iteration'].size == 0 else df['iteration'].max() + 1

        # break the loop if we've hit the desired number of initial samples
        if i > initial_sample_count: break

        # create the dictionary for this iteration
        row_dict = {}

        # add values for all the fixed parameters
        for key in fixed_parameters:
            row_dict[key] = fixed_parameter_values[key]

        # sample values for all the free parameters
        for key in free_parameters:
            row_dict[key] = uniform_sample(optimisation_bounds[key])

        # produce transport profiles defined by new point
        radius = profile_radius_axis()

        chi_params = [row_dict[k] for k in conductivity_profile]
        chi = linear_transport_profile(radius, chi_params)

        D_params = [row_dict[k] for k in diffusivity_profile]
        D = linear_transport_profile(radius, D_params)
        dna = row_dict['D_div']
        hc  = row_dict['chi_div']

        logging.info(f"--- Starting iteration {i} ---")
        logging.info('New chi parameters:')
        logging.info(chi_params)
        logging.info('New D parameters:')
        logging.info(D_params)
        logging.info('Divertor parameters:')
        logging.info([dna, hc])

        # Run SOLPS for the new point
        run_status = run_solps(
            chi=chi, chi_r=radius, D=D, D_r=radius, iteration=i, dna=dna, hci=hc,
            hce=hc, run_directory=run_directory, output_directory=output_directory,
            solps_n_timesteps=solps_n_timesteps, solps_dt=solps_dt,
            timeout_hours=solps_timeout_hours, n_proc=solps_n_proc,
            n_species=solps_n_species, set_div_transport=set_divertor_transport
        )

        if run_status == False:
            logging.info('[initial_sampling] Restoring SOLPS run directory from reference.')
            reset_solps(run_directory, ref_directory)
            logging.info('[initial_sampling] Restoration complete, trying new run...')
            continue

        # evaluate the log-probabilities
        logprobs = evaluate_log_posterior(
            iteration=i,
            directory=output_directory,
            diagnostic_data_files=diagnostic_data_files,
            diagnostic_data_observables=diagnostic_data_observables,
            diagnostic_data_errors=diagnostic_data_errors
        )

        gaussian_logprob, cauchy_logprob, laplace_logprob, logistic_logprob = logprobs

        # build a new row for the dataframe
        row_dict['iteration'] = i
        row_dict['gaussian_logprob'] = gaussian_logprob
        row_dict['cauchy_logprob'] = cauchy_logprob
        row_dict['laplace_logprob'] = laplace_logprob
        row_dict['logistic_logprob'] = logistic_logprob
        row_dict['prediction_mean'] = None
        row_dict['prediction_error'] = None
        row_dict['convergence_metric'] = None

        df.loc[i] = row_dict  # add the new row
        df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data

        if i % solps_iter_reset == 0:
            reset_solps(run_directory, ref_directory)


def optimizer(settings_filepath):
    """
    Performs Gaussian-process optimization to maximise agreement
    between SOLPS and the given experimental data.

    :param settings_filepath: The path to the settings file.
    :return:
    """

    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # Check other data files are present
    check_dependencies(settings)

    # set-up the log file
    logger_setup(settings_filepath)

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
    error_model = settings['error_model']
    covariance_kernel_class = settings['covariance_kernel']
    trust_region = settings['trust_region']
    trust_region_width = settings['trust_region_width']
    log_scale_bounds = settings['log_scale_bounds']


    all_parameters = [key for key in fixed_parameter_values.keys()]
    free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]


    # start the optimisation loop
    while True:
        # load the training data
        df = read_hdf(output_directory + training_data_file, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            break

        # get the current iteration number
        i = df['iteration'].max()+1
        logging.info(f"--- Starting iteration {i} ---")

        # extract the training data
        logprob_key = check_error_model(error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        parameters = []
        for tup in zip(*[df[p] for p in free_parameters]):
            parameters.append( array(tup) )

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [optimisation_bounds[k] for k in free_parameters]
        normalised_parameters = [bounds_transform(p, free_parameter_bounds) for p in parameters]

        # build the set of grid-transformed points
        grid_set = set( grid_transform(p) for p in normalised_parameters )

        # set the covariance kernel parameter bounds
        amplitude = log(log_posterior.ptp())
        hyperpar_bounds = [(amplitude-3, amplitude+3)]
        hyperpar_bounds.extend( [log_scale_bounds for _ in free_parameters] )

        # construct the GP
        covariance_kernel = covariance_kernel_class(hyperpar_bounds=hyperpar_bounds)
        GP = GpRegressor(
            normalised_parameters,
            log_posterior,
            cross_val=cross_validation,
            kernel=covariance_kernel,
            optimizer="diffev"
        )
        bfgs_hps = GP.multistart_bfgs(starts=300, n_processes=solps_n_proc)

        if GP.model_selector(bfgs_hps) > GP.model_selector(GP.hyperpars):
            mode = bfgs_hps
        else:
            mode = GP.hyperpars

        logging.info('[optimiser] GP hyper-parameter tuning complete - hyper-parameter values are:')
        logging.info(mode)

        # If a trust-region approach is being used, limit the search area
        # to a region around the current maximum
        trhw = 0.5*trust_region_width
        if trust_region:
            max_ind = log_posterior.argmax()
            max_point = normalised_parameters[max_ind]
            search_bounds = [(max(0., v-trhw), min(1., v+trhw)) for v in max_point]
        else:
            search_bounds = [(0.,1.) for i in range(len(free_parameters))]

        # build the GP-optimiser
        covariance_kernel = covariance_kernel_class(hyperpar_bounds=hyperpar_bounds)
        GPopt = GpOptimiser(
            normalised_parameters,
            log_posterior,
            hyperpars=GP.hyperpars,
            bounds=search_bounds,
            cross_val=cross_validation,
            kernel=covariance_kernel,
            acquisition=acquisition_function
        )

        # maximise the acquisition both by multi-start bfgs and differential evolution,
        # and use the best of the two
        # TODO - propose_evaluation doesnt have arguments anymore - add them back
        # TODO - The way the acquis funcs select starting points for bfgs doesnt really work properly for trust regions
        bfgs_prop = GPopt.propose_evaluation(bfgs=True)
        diff_prop = GPopt.propose_evaluation()

        bfgs_acq = GPopt.acquisition(bfgs_prop)
        diff_acq = GPopt.acquisition(diff_prop)

        new_point = bfgs_prop if bfgs_acq > diff_acq else diff_prop

        logging.info('[optimiser] Acquisition function maximisation complete - max function value was:')
        logging.info(max(bfgs_acq, diff_acq))

        # calculate the convergence metric
        convergence = GPopt.acquisition.convergence_metric(new_point)

        # get predicted log-probability at the new point
        mu_lp, sigma_lp = GPopt.gp(new_point)

        # back-transform to get the new point as model parameters
        new_parameters = bounds_transform(new_point, free_parameter_bounds, inverse=True)

        # create the dictionary for this iteration
        row_dict = {}

        # add values for all the fixed parameters
        for key in fixed_parameters:
            row_dict[key] = fixed_parameter_values[key]

        # add the new free parameter values
        for key, val in zip(free_parameters, new_parameters):
            row_dict[key] = val

        # check to see if the grid-transformed new point is already in the evaluated set
        if grid_transform(new_point) in grid_set:
            raise ValueError(
                """
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                The latest proposed evaluation is a point which has been
                previously evaluated - this may indicate that a local
                maximum has been reached.
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                """
            )

        # produce transport profiles defined by new point
        radius = profile_radius_axis()

        chi_params = [row_dict[k] for k in conductivity_profile]
        chi = linear_transport_profile(radius, chi_params)

        D_params = [row_dict[k] for k in diffusivity_profile]
        D = linear_transport_profile(radius, D_params)
        dna = row_dict['D_div']
        hc  = row_dict['chi_div']

        logging.info('New chi parameters:')
        logging.info(chi_params)
        logging.info('New D parameters:')
        logging.info(D_params)
        logging.info('Divertor parameters:')
        logging.info([dna, hc])

        # Run SOLPS for the new point
        run_status = run_solps(
            chi=chi, chi_r=radius, D=D, D_r=radius, iteration=i, dna=dna, hci=hc,
            hce=hc, run_directory=run_directory, output_directory=output_directory,
            solps_n_timesteps=solps_n_timesteps, solps_dt=solps_dt,
            timeout_hours=solps_timeout_hours, n_proc=solps_n_proc,
            n_species=solps_n_species, set_div_transport=set_divertor_transport
        )

        if run_status == False:
            logging.info('[optimiser] Restoring SOLPS run directory from reference.')
            reset_solps(run_directory, ref_directory)
            logging.info('[optimiser] Restoration complete, trying new run...')
            continue

        # evaluate the chi-squared
        logprobs = evaluate_log_posterior(
            iteration=i,
            directory=output_directory,
            diagnostic_data_files=diagnostic_data_files,
            diagnostic_data_observables=diagnostic_data_observables,
            diagnostic_data_errors=diagnostic_data_errors
        )

        gaussian_logprob, cauchy_logprob, laplace_logprob, logistic_logprob = logprobs

        # build a new row for the dataframe
        row_dict['iteration'] = i
        row_dict['gaussian_logprob'] = gaussian_logprob
        row_dict['cauchy_logprob'] = cauchy_logprob
        row_dict['laplace_logprob'] = laplace_logprob
        row_dict['logistic_logprob'] = logistic_logprob
        row_dict['prediction_mean'] = mu_lp,
        row_dict['prediction_error'] = sigma_lp,
        row_dict['convergence_metric'] = convergence

        df.loc[df.index.max()+1] = row_dict  # add the new row
        df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data
        del df

        if i % solps_iter_reset == 0:
            reset_solps(run_directory, ref_directory)







def random_search(settings_filepath):
    """
    Performs random search (within a trust-region if requested) to
    maximise agreement between SOLPS and the given experimental data.

    :param settings_filepath: The path to the settings file.
    """

    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # Check other data files are present
    check_dependencies(settings)

    # set-up the log file
    logger_setup(settings_filepath)

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
    error_model = settings['error_model']
    covariance_kernel_class = settings['covariance_kernel']
    trust_region = settings['trust_region']
    trust_region_width = settings['trust_region_width']
    log_scale_bounds = settings['log_scale_bounds']


    all_parameters = [key for key in fixed_parameter_values.keys()]
    free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]


    # start the optimisation loop
    while True:
        # load the training data
        df = read_hdf(output_directory + training_data_file, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            break

        # get the current iteration number
        i = df['iteration'].max()+1
        logging.info(f"--- Starting iteration {i} ---")

        # extract the training data
        logprob_key = check_error_model(error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        parameters = []
        for tup in zip(*[df[p] for p in free_parameters]):
            parameters.append( array(tup) )

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [optimisation_bounds[k] for k in free_parameters]
        normalised_parameters = [bounds_transform(p, free_parameter_bounds) for p in parameters]

        # build the set of grid-transformed points
        grid_set = {grid_transform(p) for p in normalised_parameters}

        # If a trust-region approach is being used, limit the search area
        # to a region around the current maximum
        trhw = 0.5*trust_region_width
        if trust_region:
            max_ind = log_posterior.argmax()
            max_point = normalised_parameters[max_ind]
            search_bounds = [(max(0., v-trhw), min(1., v+trhw)) for v in max_point]
        else:
            search_bounds = [(0.,1.) for i in range(len(free_parameters))]


        new_point = array([a + random()*(b-a) for a,b in search_bounds])

        # calculate the convergence metric
        convergence = 0.

        # get predicted log-probability at the new point
        mu_lp, sigma_lp = 0, 0

        # back-transform to get the new point as model parameters
        new_parameters = bounds_transform(new_point, free_parameter_bounds, inverse=True)

        # create the dictionary for this iteration
        row_dict = {}

        # add values for all the fixed parameters
        for key in fixed_parameters:
            row_dict[key] = fixed_parameter_values[key]

        # add the new free parameter values
        for key, val in zip(free_parameters, new_parameters):
            row_dict[key] = val

        # produce transport profiles defined by new point
        radius = profile_radius_axis()

        chi_params = [row_dict[k] for k in conductivity_profile]
        chi = linear_transport_profile(radius, chi_params)

        D_params = [row_dict[k] for k in diffusivity_profile]
        D = linear_transport_profile(radius, D_params)
        dna = row_dict['D_div']
        hc  = row_dict['chi_div']

        logging.info('New chi parameters:')
        logging.info(chi_params)
        logging.info('New D parameters:')
        logging.info(D_params)
        logging.info('Divertor parameters:')
        logging.info([dna, hc])

        # Run SOLPS for the new point
        run_status = run_solps(
            chi=chi,
            chi_r=radius,
            D=D,
            D_r=radius,
            iteration=i,
            dna=dna,
            hci=hc,
            hce=hc,
            run_directory=run_directory,
            output_directory=output_directory,
            solps_n_timesteps=solps_n_timesteps,
            solps_dt=solps_dt,
            timeout_hours=solps_timeout_hours,
            n_proc=solps_n_proc,
            n_species=solps_n_species,
            set_div_transport=set_divertor_transport
        )

        if run_status == False:
            logging.info('[optimiser] Restoring SOLPS run directory from reference.')
            reset_solps(run_directory,ref_directory)
            logging.info('[optimiser] Restoration complete, trying new run...')
            continue

        # evaluate the chi-squared
        logprobs = evaluate_log_posterior(
            iteration=i,
            directory=output_directory,
            diagnostic_data_files=diagnostic_data_files,
            diagnostic_data_observables=diagnostic_data_observables,
            diagnostic_data_errors=diagnostic_data_errors
        )

        gaussian_logprob, cauchy_logprob, laplace_logprob, logistic_logprob = logprobs

        # build a new row for the dataframe
        row_dict['iteration'] = i
        row_dict['gaussian_logprob'] = gaussian_logprob
        row_dict['cauchy_logprob'] = cauchy_logprob
        row_dict['laplace_logprob'] = laplace_logprob
        row_dict['logistic_logprob'] = logistic_logprob
        row_dict['prediction_mean'] = mu_lp,
        row_dict['prediction_error'] = sigma_lp,
        row_dict['convergence_metric'] = convergence

        df.loc[df.index.max()+1] = row_dict  # add the new row
        df.to_hdf(output_directory + training_data_file, key='training', mode='w')  # save the data
        del df

        if i % solps_iter_reset == 0:
            reset_solps(run_directory, ref_directory)
