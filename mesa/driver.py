from mesa.simulations import Simulation, SimulationRun
from mesa.diagnostics import WeightedObjectiveFunction
from pandas import read_hdf
from inference.pdf import BinaryTree
from numpy.random import default_rng
from numpy import array, ndarray
from inference.gp import GpRegressor, GpOptimiser
from abc import ABC, abstractmethod
from time import sleep
import subprocess
import logging
import os

class Driver(ABC):
    """
    Base class for all drivers, holds list of parameters and their
    limits in a single dictionary. Also holds the simulation
    object that it will launch with parameter values
    """
    simulation: Simulation
    ojective_function: WeightedObjectiveFunction
    concurrent_runs: int
    initial_sample_count: int
    max_iterations: int
    trainingfile: str
    reference_dir: str
    parameter_keys = []

    # define dictionaries of parameters
    fixed_parameter_values = {}
    optimization_bounds = {}
    free_parameter_keys = []
    fixed_parameter_keys = []

    def __init__(self, params, initial_sample_count, concurrent_runs, max_iterations):
        self.parameters = params
        self.initial_sample_count = initial_sample_count
        self.concurrent_runs = concurrent_runs
        self.max_iterations = max_iterations
        self.__parse_params()
        self.rng = default_rng()

    def initialize(self, sim:Simulation, objfn:WeightedObjectiveFunction, trainingfile:str):
        self.simulation = sim
        self.objective_function = objfn
        self.trainingfile = trainingfile
        self.reference_dir = os.path.dirname(os.path.abspath(self.trainingfile))
        
    @abstractmethod
    def get_next_points(self):
        pass

    def check_error_model(self, error_model):
        if error_model.lower() in {'gaussian', 'cauchy', 'laplace', 'logistic'}:
            return error_model.lower() + '_logprob'
        else:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> The 'error_model' settings variable was specified as {error_model},
                >> but must be either 'gaussian', 'cauchy', 'laplace' or 'logistic'.
                """
            )

    def run(self, new_points=None):
        df = read_hdf(self.trainingfile, 'training')
        while not self.converged():
            current_runs = set()
            # get the current iteration number
            itr = df['iteration'].max() + 1
            if itr > self.max_iterations:
                logging.info('maximum iterations reached without convergence')
                break
            logging.info(f"--- Starting iteration {itr} ---")

            # get next set of points for this iteration
            if (new_points==None):
                new_points = self.get_next_points()

            # Run simulation for the new point
            for counter,point in enumerate(new_points):
                logging.info(f"Sub-iteration {counter} - New parameters:")
                logging.info([point[k] for k in self.free_parameter_keys])
                thisrun = self.simulation.launch(
                    iteration = itr+'_'+str(counter),
                    directory = self.simpath,
                    parameters = point
                )
                # store the iteration, launch time and parameters for the new run
                current_runs.add(thisrun)

                # check statuses if we have hit the max concurent runs or if we are on the 
                #  last of this set of points
                if (len(current_runs) >= self.concurrent_runs or counter == len(new_points)-1):
                    while len(current_runs) > 0 or len(current_runs) > 0:
                        # loop through all currently running jobs to check if
                        #  they have finished or timed-out
                        current_runs_iterable = [run for run in current_runs]
                        run : SimulationRun
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
                                    "iteration": run.iteration,
                                    "prediction_mean": None,
                                    "prediction_error": None,
                                    "prediction_metric": None
                                }
                                new_row.update(logprobs)
                                new_row.update(run.parameters)
                                df = read_hdf(self.trainingfile, 'training')
                                df.loc[(itr,counter)] = new_row  # add the new row
                                df.to_hdf(self.trainingfile, key='training', mode='w')  # save the data

                                # now the run results are saved we can stop tracking the run
                                current_runs.remove(run)

                                # clean up the run directory
                                run.cleanup()

                            elif run_status == "crashed":
                                logging.info("[ crash warning ]")
                                logging.info(f">> iteration {run.iteration}, job {run.run_id} has crashed")
                                current_runs.remove(run)  # remove it from the current runs
                                subprocess.run(["rm", "-r", run.directory])  # remove its run directory
                                current_runs_iterable.append(run.iteration)  # add the iteration number back to the queue

                            elif run_status == "timed-out":
                                logging.info("[ time-out warning ]")
                                logging.info(f">> iteration {run.iteration}, job {run.run_id} has timed-out")
                                run.cancel()  # cancel the timed-out job
                                current_runs.remove(run)  # remove it from the current runs
                                subprocess.run(["rm", "-r", run.directory])  # remove its run directory
                                current_runs_iterable.append(run.iteration)  # add the iteration number back to the queue

                        # if we're already at maximum concurrent runs, pause for a bit before re-checking
                        if len(current_runs) == self.concurrent_runs:
                            sleep(30)

    def get_dataframe_columns(self):
        """
        Returns a combination of columns for the pandas dataFrame and
        data file. This consists of columns needed by the optimizer plus
        columns of all the parameters. The optimization columns are known
        by the derived driver, while the parameter keys are read from
        the input file by the base driver class.
        """
        opt_cols = self.__get_opt_columns()
        cols = [
            'iteration',
            *opt_cols,
            *self.parameter_keys
        ]
        return cols
    
    def normalise_parameters(self, v, bounds):
        return array([(k-b[0])/(b[1]-b[0]) for k, b in zip(v, bounds)])

    def reverse_normalisation(self, v, bounds):
        return array([b[0] + (b[1] - b[0]) * k for k, b in zip(v, bounds)])

    def grid_transform(self, point):
        tree = BinaryTree(limits=(0., 1.), layers=6)
        return tuple([tree.lookup(v)[2] for v in point])

    def hypercube_sample(self, bounds):
        return [b[0] + (b[1]-b[0])*self.rng.random() for b in bounds]

    def uniform_sample(self, bounds):
        return bounds[0] + (bounds[1]-bounds[0])*self.rng.random()
    
    def __parse_params(self):
        """
        Method to parse the input parameters. Need to check which are fixed
        and which are not and separate them
        """
        self.parameter_keys = [key for key in self.parameters.keys()]
        # find which parameters are tuples of limits and which are a single number (fixed)
        for key in self.parameter_keys:
            if isinstance(self.parameters[key],tuple):
                self.optimization_bounds[key] = self.parameters[key]
                self.free_parameter_keys.append(key)
            elif isinstance(self.parameters[key],float) or isinstance(self.parameters[key],int):
                self.fixed_parameter_values[key] = self.parameters[key]
                self.fixed_parameter_keys.append(key)
            else:
                raise ValueError(
                    f"""
                    [ MESA ERROR ]
                    >> Unrecognized value in parameter list. For a free
                    >> parameter bounds must be a tuple of two numbers.
                    >> For fixed parameters the value must be float or int.
                    """
                )
        self.num_free_parameters = len(self.free_parameter_keys)

class Optimizer(Driver):
    def __init__(self, 
        params, 
        initial_sample_count=0, 
        max_iterations=200,
        concurrent_runs=1
    ):
        super().__init__(params,initial_sample_count,concurrent_runs,max_iterations)

class GPOptimizer(Optimizer):

    def __init__(self, 
        params, 
        initial_sample_count=20, 
        max_iterations=200,
        concurrent_runs = 1,
        covariance_kernel = None,
        mean_function = None,
        acquisition_function = None,
        cross_validation = False,
        error_model = 'cauchy',
        trust_region_width = 0.3
    ):
        super().__init__(self, 
            params, 
            initial_sample_count=initial_sample_count, 
            max_iterations=max_iterations,
            concurrent_runs=concurrent_runs
        )
        self.covariance_kernel = covariance_kernel
        self.mean_function = mean_function
        self.acquisition_function = acquisition_function
        self.cross_validation = cross_validation
        self.error_model = error_model
        self.trust_region_width = trust_region_width

    def initialize(self, sim, objfn, trainingfile):
        super().initialize(sim, objfn, trainingfile)
        df = read_hdf(self.trainingfile, 'training')
        current_iterations = 0 if df['iteration'].size == 0 else df['iteration'].max()
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
        iteration_queue = [i for i in range(current_iterations + 1, self.initial_sample_count + 1)][::-1]

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
                    iteration=i,
                    directory=self.reference_dir,
                    parameters=row_dict
                )

                # store the iteration, launch time and parameters for the new run
                current_runs.add(RunObj)

            # loop through all currently running jobs to check if
            # they have finished or timed-out
            current_runs_iterable = [run for run in current_runs]
            run : SimulationRun
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
                        "iteration": run.iteration,
                        "prediction_mean": None,
                        "prediction_error": None,
                        "prediction_metric": None
                    }
                    new_row.update(logprobs)
                    new_row.update(run.parameters)
                    df = read_hdf(self.trainingfile, 'training')
                    df.loc[(run.iteration,0)] = new_row  # add the new row
                    df.to_hdf(self.trainingfile, key='training', mode='w')  # save the data

                    # now the run results are saved we can stop tracking the run
                    current_runs.remove(run)

                    # clean up the run directory
                    run.cleanup()

                elif run_status == "crashed":
                    logging.info("[ crash warning ]")
                    logging.info(f">> iteration {run.iteration}, job {run.run_id} has crashed")
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove its run directory

                elif run_status == "timed-out":
                    logging.info("[ time-out warning ]")
                    logging.info(f">> iteration {run.iteration}, job {run.run_id} has timed-out")
                    run.cancel()  # cancel the timed-out job
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove its run directory

            # if we're already at maximum concurrent runs, pause for a bit before re-checking
            if len(current_runs) == self.concurrent_runs:
                sleep(30)

    def get_next_points(self):
        # load the training data
        df = read_hdf(self.trainingfile, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= self.max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            return None

        # extract the training data
        logprob_key = self.check_error_model(self.error_model)
        log_posterior = df[logprob_key].to_numpy().copy()

        # build a list of numpy arrays containing all the parameter values
        parameters = [array(t) for t in zip(*[df[k] for k in self.free_parameter_keys])]

        # convert the data to the normalised coordinates:
        free_parameter_bounds = [self.optimization_bounds[k] for k in self.free_parameter_keys]
        normalised_parameters = [self.normalise_parameters(p, free_parameter_bounds) for p in parameters]

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
            trust_region_width=self.trust_region_width
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
        
    def __propose_evaluation(self,
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
            n_starts=300
        )

        # If a trust-region approach is being used, limit the search area
        # to a region around the current maximum
        if trust_region_width is not None:
            trhw = 0.5 * trust_region_width
            max_ind = log_posterior.argmax()
            max_point = normalised_parameters[max_ind]
            search_bounds = [(max(0., v - trhw), min(1., v + trhw)) for v in max_point]
        else:
            search_bounds = [(0., 1.) for _ in range(normalised_parameters[0].size)]

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
            n_processes=n_procs
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
            "prediction_metric": convergence
        }

        return new_point, metrics

    def __get_opt_columns():
        """
        Returns the columns needed in the data file for GPO
        """
        # define what columns will be in the dataframe specific to the optimiation technique
        cols = [
            'gaussian_logprob',
            'cauchy_logprob',
            'laplace_logprob',
            'logistic_logprob',
            'prediction_mean',
            'prediction_error',
            'convergence_metric'
        ]
        return cols

class GeneticOptimizer(Optimizer):

    population : list
    generations : list

    def __init__(self, 
        params : dict, 
        initial_sample_count=20, 
        max_iterations=200,
        pop_size=8,
        tolerance=1e-8
    ):
        super().__init__( 
            params, 
            initial_sample_count=initial_sample_count, 
            max_iterations=max_iterations,
            concurrent_runs=1
        )
        self.pop_size=pop_size
        self.current_generation = 0
        self.population = []
        self.generations = []
    
    def breed(self,p1,p2):
        return
    
    def initialize(self, sim, objfn, trainingfile):
        """
        Run initial population randomly distributed
        """
        super().initialize(sim, objfn, trainingfile)
        for i in range(self.pop_size):
            individual = {}
            # select random value in bounds for free params
            for key in self.free_parameter_keys:
                bnds = self.optimization_bounds[key]
                individual[key] = (bnds[1]-bnds[0])*self.rng.random()+bnds[0]
            # add the fixed values
            for key in self.fixed_parameter_keys:
                individual[key] = self.fixed_parameter_values[key]
            self.population.append(individual)
        self.generations.append(self.population)

        # run the simulations
        self.run(new_points=self.population)

        return
    
    def get_next_points(self):
        """
        Get next generation
        """
        # load the training data
        df = read_hdf(self.trainingfile, 'training')
        # break the loop if we've hit the max number of iterations
        if df['iteration'].max() >= self.max_iterations:
            logging.info('[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations')
            return None

        # extract the training data
        fom = df['logprob']

        lastpop = self.generations[-1]
        for i in range(self.pop_size):
            # sort last population by FoM
            # remove all but top 20%
            # interbreed that 20%, adding a mutation every now and then
            self.population[i] = lastpop[i]

        self.generations.append(self.population)

        logging.info("New population:")
        logging.info([[param_dict[k] for k in self.free_parameter_keys] for param_dict in self.population])

        return self.generations[-1]

    def __get_opt_columns():
        """
        Returns the columns needed in the data file for GPO
        """
        # define what columns will be in the dataframe specific to the optimiation technique
        cols = [
            'logprob'
        ]
        return cols

#class GradientDescent(Optimizer):