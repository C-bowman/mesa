
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import array, log
from numpy.random import random
from pandas import DataFrame, read_hdf

import subprocess
from time import time, sleep
from os.path import isfile
import logging

from mesa.parsing import parse_inputs, logger_setup, check_error_model
from mesa.models import linear_transport_profile, profile_radius_axis
from mesa.optimisation import propose_evaluation
from mesa.solps import launch_solps, evaluate_log_posterior
from mesa.parameters import conductivity_profile, diffusivity_profile

from inference.pdf import BinaryTree


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
    """
    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # set-up the log file
    logger_setup(settings_filepath)

    # data & results filepaths
    reference_directory = settings['solps_ref_directory']
    training_data_file = settings['training_data_file']
    diagnostics = settings['diagnostics']

    # SOLPS settings
    solps_n_proc = settings['solps_n_proc']
    set_divertor_transport = settings['set_divertor_transport']
    concurrent_runs = settings['concurrent_runs']
    solps_timeout_hours = settings['solps_timeout_hours']
    transport_profile_bounds = settings['transport_profile_bounds']

    # optimiser settings
    fixed_parameter_values = settings['fixed_parameter_values']
    optimisation_bounds = settings['optimisation_bounds']
    initial_sample_count = settings['initial_sample_count']

    all_parameter_keys = [key for key in fixed_parameter_values.keys()]
    free_parameter_keys = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameter_keys = [key for key, value in fixed_parameter_values.items() if value is not None]

    # first check if the training data file exists already, or needs to be created
    if isfile(reference_directory + training_data_file):
        df = read_hdf(reference_directory + training_data_file, 'training')
        current_iterations = 0 if df['iteration'].size == 0 else df['iteration'].max()
        if current_iterations >= initial_sample_count:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> An initial sampling count of {initial_sample_count} was specified
                >> in the settings file, but the training data file
                >> already contains {current_iterations} iterations.
                """
            )
        # build a queue of the additional iterations which are required
        iteration_queue = [i for i in range(current_iterations + 1, initial_sample_count + 1)][::-1]
    else:
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
            *all_parameter_keys
        ]

        # create the empty dataframe to store the training data and save it to HDF
        df = DataFrame(columns=cols)
        df.to_hdf(reference_directory + training_data_file, key='training', mode='w')
        del df
        iteration_queue = [i for i in range(1, initial_sample_count + 1)][::-1]

    # loop until enough samples have been evaluated
    current_runs = set()
    while len(iteration_queue) > 0 or len(current_runs) > 0:

        # if the current number of runs is less than the allowed
        # maximum, then launch another
        if len(current_runs) < concurrent_runs and len(iteration_queue) > 0:
            i = iteration_queue.pop()

            # create the dictionary for this iteration
            row_dict = {}
            # add values for all the fixed parameters
            for key in fixed_parameter_keys:
                row_dict[key] = fixed_parameter_values[key]
            # sample values for all the free parameters
            for key in free_parameter_keys:
                row_dict[key] = uniform_sample(optimisation_bounds[key])

            logging.info(f"--- Starting iteration {i} ---")
            logging.info('New chi parameters:')
            logging.info([row_dict[k] for k in conductivity_profile])
            logging.info('New D parameters:')
            logging.info([row_dict[k] for k in diffusivity_profile])

            # Run SOLPS for the new point
            RunObj = launch_solps(
                iteration=i,
                reference_directory=reference_directory,
                parameter_dictionary=row_dict,
                transport_profile_bounds=transport_profile_bounds,
                n_proc=solps_n_proc,
                set_div_transport=set_divertor_transport
            )

            # store the iteration, launch time and parameters for the new run
            current_runs.add(RunObj)

        # loop through all currently running jobs to check if
        # they have finished or timed-out
        current_runs_iterable = [run for run in current_runs]
        for Run in current_runs_iterable:
            runtime_hours = (Run.launch_time - time()) / 3600

            run_status = Run.status()
            if run_status == "complete":
                # evaluate the log-probabilities
                logprobs = evaluate_log_posterior(
                    directory=Run.directory,
                    diagnostics=diagnostics
                )

                # build a new row for the dataframe
                new_row = {
                    "iteration": Run.iteration,
                    "prediction_mean": None,
                    "prediction_error": None,
                    "prediction_metric": None
                }
                new_row.update(logprobs)
                new_row.update(Run.parameters)
                df = read_hdf(reference_directory + training_data_file, 'training')
                df.loc[Run.iteration] = new_row  # add the new row
                df.to_hdf(reference_directory + training_data_file, key='training', mode='w')  # save the data

                # now the run results are saved we can stop tracking the run
                current_runs.remove(Run)

                # clean up the run directory
                output_files = [f for f in os.listdir(Run.directory) if isfile(Run.directory + f)]
                allowed_files = [
                    "balance.nc", "input.dat", "b2.neutrals.parameters",
                    "b2.boundary.parameters", "b2.numerics.parameters",
                    "b2.transport.parameters", "b2.transport.inputfile",
                    "b2mn.dat"
                ]
                deletions = [f for f in output_files if f not in allowed_files]
                [os.remove(Run.directory + f) for f in deletions]

            elif run_status == "crashed":
                logging.info("[ crash warning ]")
                logging.info(f">> iteration {Run.iteration}, job {Run.run_id} has crashed")
                current_runs.remove(Run)  # remove it from the current runs
                subprocess.run(["rm", "-r", Run.directory])  # remove its run directory
                iteration_queue.append(Run.iteration)  # add the iteration number back to the queue

            elif run_status == "running" and runtime_hours >= solps_timeout_hours:
                logging.info("[ time-out warning ]")
                logging.info(f">> iteration {Run.iteration}, job {Run.run_id} has timed-out")
                Run.cancel()  # cancel the timed-out job
                current_runs.remove(Run)  # remove it from the current runs
                subprocess.run(["rm", "-r", Run.directory])  # remove its run directory
                iteration_queue.append(Run.iteration)  # add the iteration number back to the queue

        # if we're already at maximum concurrent runs, pause for a bit before re-checking
        if len(current_runs) == concurrent_runs:
            sleep(30)


def optimizer(settings_filepath):
    """
    Performs Gaussian-process optimization to maximise agreement
    between SOLPS and the given experimental data.

    :param settings_filepath: The path to the settings file.
    :return:
    """

    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath, check_training_data=True)

    # set-up the log file
    logger_setup(settings_filepath)

    # data & results filepaths
    reference_directory = settings['solps_ref_directory']
    training_data_file = settings['training_data_file']
    diagnostics = settings['diagnostics']

    # SOLPS settings
    solps_n_proc = settings['solps_n_proc']
    set_divertor_transport = settings['set_divertor_transport']
    solps_timeout_hours = settings['solps_timeout_hours']
    transport_profile_bounds = settings['transport_profile_bounds']

    # optimiser settings
    max_iterations = settings['max_iterations']
    fixed_parameter_values = settings['fixed_parameter_values']
    optimisation_bounds = settings['optimisation_bounds']
    error_model = settings['error_model']
    log_scale_bounds = settings['log_scale_bounds']


    all_parameter_keys = [key for key in fixed_parameter_values.keys()]
    free_parameter_keys = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameter_keys = [key for key, value in fixed_parameter_values.items() if value is not None]


    # start the optimisation loop
    while True:
        # load the training data
        df = read_hdf(reference_directory + training_data_file, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            break

        # get the current iteration number
        itr = df['iteration'].max()+1
        logging.info(f"--- Starting iteration {itr} ---")

        # extract the training data
        logprob_key = check_error_model(error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        # build a list of numpy arrays containing all the parameter values
        parameters = [array(t) for t in zip(*[df[k] for k in free_parameter_keys])]

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [optimisation_bounds[k] for k in free_parameter_keys]
        normalised_parameters = [bounds_transform(p, free_parameter_bounds) for p in parameters]

        # build the set of grid-transformed points
        grid_set = {grid_transform(p) for p in normalised_parameters}

        # set the covariance kernel parameter bounds
        amplitude = log(log_posterior.ptp())
        hyperpar_bounds = [(amplitude-3, amplitude+3)]
        hyperpar_bounds.extend([log_scale_bounds for _ in free_parameter_keys])

        new_point, metrics = propose_evaluation(
            log_posterior=log_posterior,
            normalised_parameters=normalised_parameters,
            hyperpar_bounds=hyperpar_bounds,
            kernel=settings["covariance_kernel"],
            mean_function=settings["mean_function"],
            acquisition=settings["acquisition_function"],
            cross_validation=settings["cross_validation"],
            n_procs=settings["solps_n_proc"],
            trust_region_width=settings["trust_region_width"]
        )

        # back-transform to get the new point as model parameters
        new_parameters = bounds_transform(new_point, free_parameter_bounds, inverse=True)

        # create the dictionary for this iteration
        param_dict = {}

        # add values for all the fixed parameters
        for key in fixed_parameter_keys:
            param_dict[key] = fixed_parameter_values[key]

        # add the new free parameter values
        for key, val in zip(free_parameter_keys, new_parameters):
            param_dict[key] = val

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

        logging.info('New chi parameters:')
        logging.info([param_dict[k] for k in conductivity_profile])
        logging.info('New D parameters:')
        logging.info([param_dict[k] for k in diffusivity_profile])

        # Run SOLPS for the new point
        RunObj = launch_solps(
            iteration=itr,
            reference_directory=reference_directory,
            parameter_dictionary=param_dict,
            transport_profile_bounds=transport_profile_bounds,
            n_proc=solps_n_proc,
            set_div_transport=set_divertor_transport
        )

        while RunObj.status() == "running":
            runtime_hours = (time() - RunObj.launch_time) / 3600
            if runtime_hours >= solps_timeout_hours:
                logging.info("[ time-out warning ]")
                logging.info(f">> iteration {RunObj.iteration}, job {RunObj.run_id} has timed-out")
            sleep(20)

        if RunObj.status() == "crashed":
            logging.info("[ crash warning ]")
            logging.info(f">> iteration {RunObj.iteration}, job {RunObj.run_id} has crashed")
            exit()

        # evaluate the chi-squared
        logprobs = evaluate_log_posterior(
            directory=RunObj.directory,
            diagnostics=diagnostics
        )

        # build a new row for the dataframe
        new_row = {"iteration": RunObj.iteration}
        new_row.update(metrics)
        new_row.update(logprobs)
        new_row.update(RunObj.parameters)

        df.loc[df.index.max()+1] = new_row  # add the new row
        df.to_hdf(reference_directory + training_data_file, key='training', mode='w')  # save the data
        del df







def random_search(settings_filepath):
    """
    Performs random search (within a trust-region if requested) to
    maximise agreement between SOLPS and the given experimental data.

    :param settings_filepath: The path to the settings file.
    """

    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath, check_training_data=True)

    # set-up the log file
    logger_setup(settings_filepath)

    # data & results filepaths
    reference_directory = settings['solps_ref_directory']
    training_data_file = settings['training_data_file']
    diagnostics = settings['diagnostics']

    # SOLPS settings
    solps_n_proc = settings['solps_n_proc']
    set_divertor_transport = settings['set_divertor_transport']
    solps_timeout_hours = settings['solps_timeout_hours']
    transport_profile_bounds = settings['transport_profile_bounds']

    # optimiser settings
    max_iterations = settings['max_iterations']
    fixed_parameter_values = settings['fixed_parameter_values']
    optimisation_bounds = settings['optimisation_bounds']
    error_model = settings['error_model']
    trust_region = settings['trust_region']
    trust_region_width = settings['trust_region_width']


    all_parameters = [key for key in fixed_parameter_values.keys()]
    free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
    fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]


    # start the optimisation loop
    while True:
        # load the training data
        df = read_hdf(reference_directory + training_data_file, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            break

        # get the current iteration number
        itr = df['iteration'].max() + 1
        logging.info(f"--- Starting iteration {itr} ---")

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
            search_bounds = [(0., 1.) for _ in range(len(free_parameters))]


        new_point = array([a + random()*(b-a) for a,b in search_bounds])

        # calculate the convergence metric
        convergence = 0.

        # get predicted log-probability at the new point
        mu_lp, sigma_lp = 0, 0

        # back-transform to get the new point as model parameters
        new_parameters = bounds_transform(new_point, free_parameter_bounds, inverse=True)

        # create the dictionary for this iteration
        param_dict = {}

        # add values for all the fixed parameters
        for key in fixed_parameters:
            param_dict[key] = fixed_parameter_values[key]

        # add the new free parameter values
        for key, val in zip(free_parameters, new_parameters):
            param_dict[key] = val

        logging.info('New chi parameters:')
        logging.info([param_dict[k] for k in conductivity_profile])
        logging.info('New D parameters:')
        logging.info([param_dict[k] for k in diffusivity_profile])

        # Run SOLPS for the new point
        RunObj = launch_solps(
            iteration=itr,
            reference_directory=reference_directory,
            parameter_dictionary=param_dict,
            transport_profile_bounds=transport_profile_bounds,
            n_proc=solps_n_proc,
            set_div_transport=set_divertor_transport
        )

        # TODO - we have no crash handling here
        while RunObj.status() == "running":
            runtime_hours = (time() - RunObj.launch_time) / 3600
            if runtime_hours >= solps_timeout_hours:
                logging.info("[ time-out warning ]")
                logging.info(f">> iteration {RunObj.iteration}, job {RunObj.run_id} has timed-out")
            sleep(20)

        # evaluate the chi-squared
        logprobs = evaluate_log_posterior(
            directory=RunObj.directory,
            diagnostics=diagnostics
        )

        # build a new row for the dataframe
        new_row = {
            "iteration": RunObj.iteration,
            "prediction_mean": mu_lp,
            "prediction_error": sigma_lp,
            "prediction_metric": convergence
        }
        new_row.update(logprobs)
        new_row.update(RunObj.parameters)

        df.loc[df.index.max()+1] = new_row  # add the new row
        df.to_hdf(reference_directory + training_data_file, key='training', mode='w')  # save the data
        del df
