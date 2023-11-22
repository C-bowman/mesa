import logging
import os
import subprocess
from abc import ABC, abstractmethod
from time import sleep

from numpy import array
from numpy.random import default_rng
from pandas import read_hdf

from mesa.diagnostics import WeightedObjectiveFunction
from mesa.simulations import Simulation, SimulationRun


class Driver(ABC):
    """
    Base class for all drivers, holds list of parameters and their
    limits in a single dictionary. Also holds the simulation
    object that it will launch with parameter values
    """

    simulation: Simulation
    objective_function: WeightedObjectiveFunction
    concurrent_runs: int
    initial_sample_count: int
    max_iterations: int
    training_file: str
    reference_dir: str
    parameter_keys = []
    converged = bool

    # define dictionaries of parameters
    fixed_parameter_values = {}
    optimization_bounds = {}
    free_parameter_keys = []
    fixed_parameter_keys = []
    opt_cols = []

    def __init__(self, params, initial_sample_count, concurrent_runs, max_iterations):
        self.parameters = params
        self.initial_sample_count = initial_sample_count
        self.concurrent_runs = concurrent_runs
        self.max_iterations = max_iterations
        self.__parse_params()
        self.rng = default_rng()
        self.converged = False

    def initialize(
        self, sim: Simulation, objfn: WeightedObjectiveFunction, training_file: str
    ):
        self.simulation = sim
        self.objective_function = objfn
        self.reference_dir = os.path.dirname(os.path.abspath(training_file))
        self.training_file = training_file.split("/")[-1]
        os.chdir(self.reference_dir)

    @abstractmethod
    def get_next_points(self):
        pass

    def check_error_model(self, error_model):
        if error_model.lower() in {"gaussian", "cauchy", "laplace", "logistic"}:
            return error_model.lower() + "_logprob"
        else:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> The 'error_model' settings variable was specified as {error_model},
                >> but must be either 'gaussian', 'cauchy', 'laplace' or 'logistic'.
                """
            )

    def run(self, new_points=None, start_iter=False):
        df = read_hdf(self.training_file, "training")
        self.fname = self.training_file.split("/")[-1]  # get just name
        while not self.converged:
            current_runs = set()
            # get the current iteration number
            if start_iter:
                itr = 0
            else:
                itr = df.index[-1][0] + 1

            if itr > self.max_iterations:
                logging.info("maximum iterations reached without convergence")
                break
            logging.info(f"--- Starting iteration {itr} ---")

            # get next set of points for this iteration
            if new_points == None:
                new_points = self.get_next_points()

            # Run simulation for the new point
            for counter, point in enumerate(new_points):
                logging.info(f"Sub-iteration {counter} - New parameters:")
                logging.info([point[k] for k in self.free_parameter_keys])
                thisrun = self.simulation.launch(
                    iteration=str(itr) + "_" + str(counter),
                    directory=self.reference_dir,
                    parameters=point,
                )
                # store the iteration, launch time and parameters for the new run
                current_runs.add(thisrun)

                # check statuses if we have hit the max concurrent runs or if we are on
                # the last of this set of points
                if (
                    len(current_runs) >= self.concurrent_runs
                    or counter == len(new_points) - 1
                ):
                    while len(current_runs) > 0 or len(current_runs) > 0:
                        # loop through all currently running jobs to check if
                        #  they have finished or timed-out
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
                                new_row = {}
                                new_row.update(logprobs)
                                new_row.update(run.parameters)
                                df = read_hdf(self.fname, "training")
                                df.loc[(itr, counter), :] = new_row  # add the new row
                                df.to_hdf(
                                    self.fname, key="training", mode="w"
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
                                subprocess.run(["rm", "-r", run.directory])  # remove its run directory
                                current_runs_iterable.append(run.iteration)  # add the iteration number back to the queue

                            elif run_status == "timed-out":
                                logging.info("[ time-out warning ]")
                                logging.info(
                                    f">> iteration {run.iteration}, job {run.run_id} has timed-out"
                                )
                                run.cancel()  # cancel the timed-out job
                                current_runs.remove(run)  # remove it from the current runs
                                subprocess.run(["rm", "-r", run.directory])  # remove its run directory
                                current_runs_iterable.append(run.iteration)  # add the iteration number back to the queue

                        # if we're already at maximum concurrent runs, pause for a bit before re-checking
                        if len(current_runs) == self.concurrent_runs:
                            sleep(30)

            # reset points so another set will be asked for next iteration
            new_points = None
            # if this is the starting iteration called from init, only do one loop iteration
            if start_iter:
                break

    def get_dataframe_columns(self):
        """
        Returns a combination of columns for the pandas dataFrame and
        data file. This consists of columns needed by the optimizer plus
        columns of all the parameters. The optimization columns are known
        by the derived driver, while the parameter keys are read from
        the input file by the base driver class.
        """
        cols = [*self.opt_cols, *self.parameter_keys]
        return cols

    def __get_opt_columns(self):
        return []

    def normalise_parameters(self, v, bounds):
        return array([(k - b[0]) / (b[1] - b[0]) for k, b in zip(v, bounds)])

    def reverse_normalisation(self, v, bounds):
        return array([b[0] + (b[1] - b[0]) * k for k, b in zip(v, bounds)])

    def grid_transform(self, point):
        tree = BinaryTree(limits=(0.0, 1.0), layers=6)
        return tuple([tree.lookup(v)[2] for v in point])

    def hypercube_sample(self, bounds):
        return [b[0] + (b[1] - b[0]) * self.rng.random() for b in bounds]

    def uniform_sample(self, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * self.rng.random()

    def __parse_params(self):
        """
        Method to parse the input parameters. Need to check which are fixed
        and which are not and separate them
        """
        self.parameter_keys = [key for key in self.parameters.keys()]
        # find which parameters are tuples of limits and which are a single number (fixed)
        for key in self.parameter_keys:
            if isinstance(self.parameters[key], tuple):
                self.optimization_bounds[key] = self.parameters[key]
                self.free_parameter_keys.append(key)
            elif isinstance(self.parameters[key], float) or isinstance(
                self.parameters[key], int
            ):
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
    def __init__(
        self, params, initial_sample_count=0, max_iterations=200, concurrent_runs=1
    ):
        super().__init__(params, initial_sample_count, concurrent_runs, max_iterations)
