import logging
import subprocess
from time import sleep

from numpy import array, ndarray
from pandas import read_hdf

from mesa.drivers import Optimizer
from mesa.simulations import SimulationRun


class GPOptimizer(Optimizer):
    def __init__(
        self,
        params,
        initial_sample_count=20,
        max_iterations=200,
        concurrent_runs=1,
        covariance_kernel=None,
        mean_function=None,
        acquisition_function=None,
        cross_validation=False,
        error_model="cauchy",
        trust_region_width=0.3,
    ):
        super().__init__(
            self,
            params,
            initial_sample_count=initial_sample_count,
            max_iterations=max_iterations,
            concurrent_runs=concurrent_runs,
        )
        self.covariance_kernel = covariance_kernel
        self.mean_function = mean_function
        self.acquisition_function = acquisition_function
        self.cross_validation = cross_validation
        self.error_model = error_model
        self.trust_region_width = trust_region_width

        self.opt_cols = [
            "gaussian_logprob",
            "cauchy_logprob",
            "laplace_logprob",
            "logistic_logprob",
            "prediction_mean",
            "prediction_error",
            "convergence_metric",
        ]

    def initialize(self, sim, objfn, trainingfile):
        super().initialize(sim, objfn, trainingfile)
        df = read_hdf(self.training_file, "training")
        current_iterations = 0 if df["iteration"].size == 0 else df["iteration"].max()
        if current_iterations >= self.initial_sample_count:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> An initial sampling count of {self.initial_sample_count} was specified
                >> in the settings file, but the training data file
                >> already contains {current_iterations} iterations.
                """
            )
        # build a queue of the additional iterations which are required
        iteration_queue = [
            i for i in range(current_iterations + 1, self.initial_sample_count + 1)
        ][::-1]

        # loop until enough samples have been evaluated
        current_runs = set()
        while len(iteration_queue) > 0 or len(current_runs) > 0:

            # if the current number of runs is less than the allowed
            # maximum, then launch another
            if len(current_runs) < self.concurrent_runs and len(iteration_queue) > 0:
                i = iteration_queue.pop()

                # create the dictionary for this iteration
                row_dict = {}
                # add values for all the fixed parameters
                for key in self.fixed_parameter_keys:
                    row_dict[key] = self.fixed_parameter_values[key]
                # sample values for all the free parameters
                for key in self.free_parameter_keys:
                    row_dict[key] = self.uniform_sample(self.optimization_bounds[key])

                logging.info(f"--- Starting iteration {i} ---")
                logging.info("New parameters:")
                logging.info([self.parameters[k] for k in self.free_parameter_keys])

                # Run simulation for the new point
                RunObj = self.simulation.launch(
                    iteration=i, directory=self.reference_dir, parameters=row_dict
                )

                # store the iteration, launch time and parameters for the new run
                current_runs.add(RunObj)

            # loop through all currently running jobs to check if
            # they have finished or timed-out
            current_runs_iterable = [run for run in current_runs]
            run: SimulationRun
            for run in current_runs_iterable:

                run_status = run.status()
                if run_status == "complete":
                    # read the simulation data
                    sim_data = self.simulation.get_data(run.directory)

                    # get the objective function value
                    logprobs = self.objective_function.evaluate(
                        simulation_interface=sim_data
                    )

                    # build a new row for the dataframe
                    new_row = {
                        "prediction_mean": None,
                        "prediction_error": None,
                        "prediction_metric": None,
                    }
                    new_row.update(logprobs)
                    new_row.update(run.parameters)
                    df = read_hdf(self.training_file, "training")
                    df.loc[(run.iteration, 0), :] = new_row  # add the new row
                    df.to_hdf(
                        self.training_file, key="training", mode="w"
                    )  # save the data

                    # now the run results are saved we can stop tracking the run
                    current_runs.remove(run)

                    # clean up the run directory
                    run.cleanup()

                elif run_status == "crashed":
                    logging.info("[ crash warning ]")
                    logging.info(
                        f">> iteration {run.iteration}, job {run.run_id} has crashed"
                    )
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(
                        ["rm", "-r", run.directory]
                    )  # remove its run directory

                elif run_status == "timed-out":
                    logging.info("[ time-out warning ]")
                    logging.info(
                        f">> iteration {run.iteration}, job {run.run_id} has timed-out"
                    )
                    run.cancel()  # cancel the timed-out job
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(
                        ["rm", "-r", run.directory]
                    )  # remove its run directory

            # if we're already at maximum concurrent runs, pause for a bit before re-checking
            if len(current_runs) == self.concurrent_runs:
                sleep(30)

    def get_next_points(self):
        # load the training data
        df = read_hdf(self.training_file, "training")
        # break the loop if we've hit the max number of iterations
        if df.index[-1][0] >= self.max_iterations:
            logging.info(
                "[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations"
            )
            return None

        # extract the training data
        logprob_key = self.check_error_model(self.error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        # build a list of numpy arrays containing all the parameter values
        parameters = [array(t) for t in zip(*[df[k] for k in self.free_parameter_keys])]

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [
            self.optimization_bounds[k] for k in self.free_parameter_keys
        ]
        normalised_parameters = [
            self.normalise_parameters(p, free_parameter_bounds) for p in parameters
        ]

        # build the set of grid-transformed points
        grid_set = {self.grid_transform(p) for p in normalised_parameters}

        # use GPO to propose a new evaluation point
        new_point, metrics = self.__propose_evaluation(
            log_posterior=log_posterior,
            normalised_parameters=normalised_parameters,
            kernel=self.covariance_kernel,
            mean_function=self.mean_function,
            acquisition=self.acquisition_function,
            cross_validation=self.cross_validation,
            n_procs=self.simulation.n_proc,
            trust_region_width=self.trust_region_width,
        )

        # back-transform to get the new point as model parameters
        new_parameters = self.reverse_normalisation(new_point, free_parameter_bounds)

        # create the dictionary for this iteration
        param_dict = {}

        # add values for all the fixed parameters
        for key in self.fixed_parameter_keys:
            param_dict[key] = self.fixed_parameter_values[key]

        # add the new free parameter values
        for key, val in zip(self.free_parameter_keys, new_parameters):
            param_dict[key] = val

        # check to see if the grid-transformed new point is already in the evaluated set
        if self.grid_transform(new_point) in grid_set:
            raise ValueError(
                """
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                The latest proposed evaluation is a point which has been
                previously evaluated - this may indicate that a local
                maximum has been reached.
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                """
            )

        logging.info("New parameters:")
        logging.info([param_dict[k] for k in self.free_parameter_keys])
        # logging.info('New chi parameters:')
        # logging.info([param_dict[k] for k in self.simulation.conductivity_profile])
        # logging.info('New D parameters:')
        # logging.info([param_dict[k] for k in self.simulation.diffusivity_profile])

        return [param_dict]

    def __propose_evaluation(
        self,
        log_posterior: ndarray,
        normalised_parameters,
        kernel,
        mean_function,
        acquisition,
        cross_validation: bool,
        n_procs: int,
        trust_region_width=None,
    ):
        GP = GpRegressor(
            normalised_parameters,
            log_posterior,
            cross_val=cross_validation,
            kernel=kernel,
            mean=mean_function,
            optimizer="bfgs",
            n_processes=n_procs,
            n_starts=300,
        )

        # If a trust-region approach is being used, limit the search area
        # to a region around the current maximum
        if trust_region_width is not None:
            trhw = 0.5 * trust_region_width
            max_ind = log_posterior.argmax()
            max_point = normalised_parameters[max_ind]
            search_bounds = [
                (max(0.0, v - trhw), min(1.0, v + trhw)) for v in max_point
            ]
        else:
            search_bounds = [(0.0, 1.0) for _ in range(normalised_parameters[0].size)]

        # build the GP-optimiser
        GPopt = GpOptimiser(
            normalised_parameters,
            log_posterior,
            hyperpars=GP.hyperpars,
            bounds=search_bounds,
            cross_val=cross_validation,
            kernel=kernel,
            mean=mean_function,
            acquisition=acquisition,
            n_processes=n_procs,
        )

        # maximise the acquisition both by multi-start bfgs and differential evolution,
        # and use the best of the two
        bfgs_prop = GPopt.propose_evaluation(optimizer="bfgs")
        diff_prop = GPopt.propose_evaluation(optimizer="diffev")

        bfgs_acq = GPopt.acquisition(bfgs_prop)
        diff_acq = GPopt.acquisition(diff_prop)

        new_point = bfgs_prop if bfgs_acq > diff_acq else diff_prop

        # calculate the convergence metric
        convergence = GPopt.acquisition.convergence_metric(new_point)

        # get predicted log-probability at the new point
        mu_lp, sigma_lp = GPopt.gp(new_point)

        metrics = {
            "prediction_mean": mu_lp,
            "prediction_error": sigma_lp,
            "prediction_metric": convergence,
        }

        return new_point, metrics
