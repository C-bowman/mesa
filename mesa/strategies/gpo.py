import logging
from numpy import array, ndarray
from pandas import read_hdf
from inference.gp import GpRegressor, GpOptimiser

from mesa.strategies import Strategy
from mesa.simulations import SimulationRun


class GPOptimizer(Strategy):
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
        self.trust_region_width = trust_region_width

        self.opt_cols = [
            "prediction_mean",
            "prediction_error",
            "convergence_metric",
        ]

    def get_initial_samples(self) -> list[dict]:
        points = []
        # create the dictionary for this iteration
        for i in range(self.initial_sample_count):
            # sample values for the free parameters
            free_params = {
                param: self.uniform_sample(bounds)
                for param, bounds in self.optimization_bounds.items()
            }

            all_params = {**free_params, **self.fixed_parameters}
            points.append(all_params)
        return points

    def get_next_points(self) -> list[dict]:
        # load the training data
        df = read_hdf(self.training_file, "training")
        # extract the training data
        objective = df["objective_value"].to_numpy().copy()

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
        new_point, metrics = self.__propose_gpo_evaluation(
            objective_values=objective,
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

        # add the new free parameter values
        free_params = {
            key: val for key, val in zip(self.free_parameter_keys, new_parameters)
        }

        param_dict = {**free_params, **self.fixed_parameters}

        # check to see if the grid-transformed new point is already in the evaluated set
        if self.grid_transform(new_point) in grid_set:
            raise ValueError(
                """\n
                \r~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                \rThe latest proposed evaluation is a point which has been
                \rpreviously evaluated - this may indicate that a local
                \rmaximum has been reached.
                \r~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                """
            )

        logging.info("New parameters:")
        logging.info([param_dict[k] for k in self.free_parameter_keys])

        return [param_dict]

    @staticmethod
    def __propose_gpo_evaluation(
        objective_values: ndarray,
        normalised_parameters,
        kernel,
        mean_function,
        acquisition,
        cross_validation: bool,
        n_procs: int,
        trust_region_width=None,
    ):
        GP = GpRegressor(
            x=normalised_parameters,
            y=objective_values,
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
            max_ind = objective_values.argmax()
            max_point = normalised_parameters[max_ind]
            search_bounds = [
                (max(0.0, v - trhw), min(1.0, v + trhw)) for v in max_point
            ]
        else:
            search_bounds = [(0.0, 1.0) for _ in range(normalised_parameters[0].size)]

        # build the GP-optimiser
        GPopt = GpOptimiser(
            x=normalised_parameters,
            y=objective_values,
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
