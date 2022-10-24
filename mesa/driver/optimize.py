from driver import Driver

class Optimizer(Driver):
    converged : bool = False
    method : str = ""
    current_state = {}

    __init__(self, sim, params):
        super().__init__(sim, params)
        self.method = method
    
    def get_current_point(self):
        return current_state

    def get_next_point(self):
        return
    
    def converged() -> bool:
        return converged
    
    def plot(self):
        return


class GPOptimizer(Optimizer):

    def initialize(self, sim):
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
                RunObj = sim.launch(
                    iteration=i,
                    reference_directory=reference_directory,
                    parameter_dictionary=row_dict,
                    transport_profile_bounds=transport_profile_bounds,
                    n_proc=solps_n_proc,
                    set_div_transport=set_divertor_transport,
                    timeout_hours=solps_timeout_hours
                )

                # store the iteration, launch time and parameters for the new run
                current_runs.add(RunObj)

            # loop through all currently running jobs to check if
            # they have finished or timed-out
            current_runs_iterable = [run for run in current_runs]
            for Run in current_runs_iterable:

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
                    Run.cleanup()

                elif run_status == "crashed":
                    logging.info("[ crash warning ]")
                    logging.info(f">> iteration {Run.iteration}, job {Run.run_id} has crashed")
                    current_runs.remove(Run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", Run.directory])  # remove its run directory
                    iteration_queue.append(Run.iteration)  # add the iteration number back to the queue

                elif run_status == "timed-out":
                    logging.info("[ time-out warning ]")
                    logging.info(f">> iteration {Run.iteration}, job {Run.run_id} has timed-out")
                    Run.cancel()  # cancel the timed-out job
                    current_runs.remove(Run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", Run.directory])  # remove its run directory
                    iteration_queue.append(Run.iteration)  # add the iteration number back to the queue

            # if we're already at maximum concurrent runs, pause for a bit before re-checking
            if len(current_runs) == concurrent_runs:
                sleep(30)

    def get_next_point(self):
        # load the training data
        df = read_hdf(reference_directory + training_data_file, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            break

        # extract the training data
        logprob_key = check_error_model(error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        # build a list of numpy arrays containing all the parameter values
        parameters = [array(t) for t in zip(*[df[k] for k in free_parameter_keys])]

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [optimisation_bounds[k] for k in free_parameter_keys]
        normalised_parameters = [normalise_parameters(p, free_parameter_bounds) for p in parameters]

        # build the set of grid-transformed points
        grid_set = {grid_transform(p) for p in normalised_parameters}

        # use GPO to propose a new evaluation point
        new_point, metrics = propose_evaluation(
            log_posterior=log_posterior,
            normalised_parameters=normalised_parameters,
            kernel=settings["covariance_kernel"],
            mean_function=settings["mean_function"],
            acquisition=settings["acquisition_function"],
            cross_validation=settings["cross_validation"],
            n_procs=solps_n_proc,
            trust_region_width=settings["trust_region_width"]
        )

        # back-transform to get the new point as model parameters
        new_parameters = reverse_normalisation(new_point, free_parameter_bounds)

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

        return param_dict

class GeneticOptimizer(Optimizer):

class GradientDescent(Optimizer):