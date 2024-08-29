import logging
import os

import subprocess
from collections.abc import Sequence
from abc import ABC, abstractmethod
from time import sleep

from numpy import array, ndarray
from numpy.random import default_rng
from pandas import read_hdf

from mesa.diagnostics import ObjectiveFunction
from mesa.simulations import Simulation, SimulationRun


class Strategy(ABC):
    """
    Base class for all strategies, holds list of parameters and their
    limits in a single dictionary. Also holds the simulation
    object that it will launch with parameter values
    """

    simulation: Simulation
    objective_function: ObjectiveFunction
    max_concurrent_runs: int
    initial_sample_count: int
    max_iterations: int
    training_file: str
    reference_dir: str
    parameter_keys = []
    converged = bool

    # define dictionaries of parameters
    fixed_parameters = {}
    optimization_bounds = {}
    free_parameter_keys = []
    opt_cols = []

    def __init__(
        self,
        params: dict,
        simulation: Simulation,
        max_concurrent_runs: int,
        max_iterations: int,
    ):
        self.parameters = params
        self.max_concurrent_runs = max_concurrent_runs
        self.max_iterations = max_iterations
        self.__parse_params()
        self.rng = default_rng()
        self.converged = False
        self.simulation = simulation
        self.objective_function = objective_func
        self.reference_dir = os.path.dirname(os.path.abspath(training_file))
        self.training_file = training_file.split("/")[-1]
        os.chdir(self.reference_dir)

    @abstractmethod
    def get_next_points(self) -> list[dict]:
        pass

    def run(self, new_points: list[dict] = None, start_iter=False):
        df = read_hdf(self.training_file, "training")

        while not self.converged:
            # get the current iteration number
            initial_run_number = 0 if start_iter else df["run_number"].max() + 1
            iteration = 0 if start_iter else df["iteration"].max() + 1

            if iteration > self.max_iterations:
                logging.info("maximum iterations reached without convergence")
                break
            logging.info(f"--- Starting iteration {iteration} ---")

            # get next set of points for this iteration
            if new_points == None:
                new_points = self.get_next_points()

            self.launch_iteration(
                iteration=iteration,
                initial_run_number=initial_run_number,
                pending_points=new_points,
            )

    def launch_iteration(
        self, iteration: int, initial_run_number: int, pending_points: list[dict]
    ):
        """
        Launches and manages all the simulations runs required to complete
        the current iteration.
        """
        current_runs: set[SimulationRun] = set()
        completed_runs: set[SimulationRun] = set()
        total_requested_runs = len(pending_points)
        pending_run_numbers = set(
            range(initial_run_number, initial_run_number + total_requested_runs)
        )

        # main monitoring loop for the simulation runs
        while len(completed_runs) < total_requested_runs:
            # if we are not at the maximum allowed number of concurrent runs
            # then launch enough to bring us to the maximum
            available_runs = min(
                self.max_concurrent_runs - len(current_runs), len(pending_run_numbers)
            )
            if available_runs:
                runs_to_launch = [
                    pending_run_numbers.pop() for _ in range(available_runs)
                ]

                for run_number in runs_to_launch:
                    point = pending_points[run_number]
                    logging.info(f"Run number {run_number} - New parameters:")
                    logging.info([point[k] for k in self.free_parameter_keys])

                    current_runs.add(
                        self.simulation.launch(
                            run_number=run_number,
                            directory=self.reference_dir,
                            parameters=point,
                        )
                    )

            # we can't modify the current_runs set while we're iterating over it,
            # so make a copy which we can iterate over
            current_runs_iterable = [run for run in current_runs]
            # loop through all currently running jobs to check if
            # they have finished or timed-out
            run: SimulationRun
            for run in current_runs_iterable:
                run_status = run.status()
                if run_status == "complete":
                    # get the objective function value
                    objective_values = self.objective_function.evaluate(
                        simulation_interface=run.get_results()
                    )

                    # build a new row for the dataframe
                    new_row = {"run_number": run.run_number, "iteration": iteration}
                    new_row.update(objective_values)
                    new_row.update(run.parameters)
                    df = read_hdf(self.training_file, "training")
                    df.loc[run.run_number, :] = new_row  # add the new row
                    df.to_hdf(self.training_file, key="training", mode="w")  # save the data

                    # now the run results are saved we can stop tracking the run
                    current_runs.remove(run)
                    completed_runs.add(run)
                    # clean up the run directory
                    run.cleanup()

                elif run_status == "crashed":
                    logging.info("[ crash warning ]")
                    logging.info(
                        f">> run #{run.run_number}, job {run.run_id} has crashed"
                    )
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove run directory
                    # currently no provision for re-starting runs, so mark it complete
                    completed_runs.add(run)

                elif run_status == "timed-out":
                    logging.info("[ time-out warning ]")
                    logging.info(
                        f">> iteration {run.run_number}, job {run.run_id} has timed-out"
                    )
                    run.cancel()  # cancel the timed-out job
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove run directory
                    # currently no provision for re-starting runs, so mark it complete
                    completed_runs.add(run)

            # if we're still at the maximum concurrent runs, pause for a bit before re-checking
            if len(current_runs) == self.max_concurrent_runs:
                sleep(30)

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

    @staticmethod
    def normalise_parameters(
        v: ndarray, bounds: Sequence[tuple[float, float]]
    ) -> ndarray:
        return array([(k - b[0]) / (b[1] - b[0]) for k, b in zip(v, bounds)])

    @staticmethod
    def reverse_normalisation(
        v: ndarray, bounds: Sequence[tuple[float, float]]
    ) -> ndarray:
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
        for param, value in self.parameters.items():
            if isinstance(value, tuple):
                self.optimization_bounds[param] = value
                self.free_parameter_keys.append(param)
            elif isinstance(value, (float, int)):
                self.fixed_parameters[param] = value
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
