from pathlib import Path
from pandas import DataFrame, read_hdf, concat
from time import sleep
import subprocess
import logging

from mesa.simulations import Simulation, SimulationRun
from mesa.strategies import Strategy
from mesa.objectives import ObjectiveFunction
from mesa.input import parse_input_module


class Mesa:
    def __init__(
        self,
        parameters: dict[str, tuple[float, float] | float],
        simulation: Simulation,
        objective_function: ObjectiveFunction,
        strategy: Strategy,
        max_concurrent_runs: int,
        max_iterations: int,
        evaluations_filepath: Path,
        simulations_directory: Path
    ):
        self.parameters = parameters
        self.simulation = simulation
        self.objective_function = objective_function
        self.strategy = strategy
        self.max_concurrent_runs = max_concurrent_runs
        self.max_iterations = max_iterations

        assert evaluations_filepath.is_file() or not evaluations_filepath.exists()
        assert simulations_directory.is_dir()
        self.evaluations_filepath = evaluations_filepath
        self.simulations_directory = simulations_directory
        self.converged = False

        # split the parameters up into those which have free or fixed values
        self.free_parameter_keys = []
        self.optimization_bounds = {}
        self.fixed_parameters = {}
        for param, value in self.parameters.items():
            if isinstance(value, tuple):
                self.free_parameter_keys.append(param)
                self.optimization_bounds[param] = value
            else:
                self.fixed_parameters[param] = value

        parameter_keys = [k for k in self.parameters.keys()]
        objective_key = self.objective_function.name

        self.metadata_keys = ["run_number", "iteration"]
        self.data_columns = [*self.metadata_keys, objective_key, *parameter_keys]
        self.data_types = {
            **{k: "int64" for k in self.metadata_keys},
            **{objective_key: "float64"},
            **{k: "float64" for k in parameter_keys}
        }

        if not self.evaluations_filepath.is_file():
            self.__initialize_evaluations_data_file()

    def run(self):
        while not self.converged:
            df: DataFrame = read_hdf(self.evaluations_filepath)
            # get the current iteration number
            initial_run_number = 0 if df.empty else df["run_number"].max() + 1
            iteration = 0 if df.empty else df["iteration"].max() + 1

            if iteration > self.max_iterations:
                logging.info("maximum iterations reached without convergence")
                break
            logging.info(f"--- Starting iteration {iteration} ---")

            # get next set of points for this iteration
            new_free_params = self.strategy.propose_evaluations(
                evaluation_data=df,
                optimization_bounds=self.optimization_bounds,
                objective_name=self.objective_function.name
            )
            # join fixed / free parameters to get the full set
            new_points = [f | self.fixed_parameters for f in new_free_params]

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
                    point = pending_points.pop()
                    logging.info(f"Run number {run_number} - New parameters:")
                    logging.info([point[k] for k in self.free_parameter_keys])

                    current_runs.add(
                        self.simulation.launch(
                            run_number=run_number,
                            simulations_directory=self.simulations_directory,
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
                        simulation_results=run.get_results()
                    )

                    # build a new row for the dataframe
                    new_row = {"run_number": run.run_number, "iteration": iteration}
                    new_row.update(objective_values)
                    new_row.update(run.parameters)

                    # build a dataframe for the latest evaluation
                    new_df = DataFrame(new_row, index=[run.run_number])
                    new_df.astype(self.data_types)

                    # update the evaluation data with the new dataframe
                    df: DataFrame = read_hdf(self.evaluations_filepath)
                    df = new_df if df.empty else concat([df, new_df])
                    df.sort_index(inplace=True)
                    df.to_hdf(self.evaluations_filepath, key="evaluations", mode="w")

                    # now the run results are saved we can stop tracking the run
                    current_runs.remove(run)
                    completed_runs.add(run)
                    # clean up the run directory
                    run.cleanup()

                elif run_status == "crashed":
                    logging.info("[ crash warning ]")
                    logging.info(
                        f">> run #{run.run_number} has crashed"
                    )
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove run directory
                    # currently no provision for re-starting runs, so mark it complete
                    completed_runs.add(run)

                elif run_status == "timed-out":
                    logging.info("[ time-out warning ]")
                    logging.info(
                        f">> iteration {run.run_number} has timed-out"
                    )
                    run.cancel()  # cancel the timed-out job
                    current_runs.remove(run)  # remove it from the current runs
                    subprocess.run(["rm", "-r", run.directory])  # remove run directory
                    # currently no provision for re-starting runs, so mark it complete
                    completed_runs.add(run)

            # if we're still at the maximum concurrent runs, pause for a bit before re-checking
            if len(current_runs) == self.max_concurrent_runs:
                sleep(5)

    def __initialize_evaluations_data_file(self):
        # create the empty dataframe to store the evaluation data and save it to HDF
        df = DataFrame(columns=self.data_columns)
        df.astype(self.data_types)
        df.to_hdf(
            self.evaluations_filepath,
            key="evaluations",
            mode="w",
        )

    @classmethod
    def build_from_input_module(cls, input_module: Path):
        mesa_inputs = parse_input_module(input_module)
        return cls(
            parameters = mesa_inputs.parameters,
            simulation = mesa_inputs.simulation,
            objective_function = mesa_inputs.objective_function,
            strategy = mesa_inputs.strategy,
            max_concurrent_runs = mesa_inputs.max_concurrent_runs,
            max_iterations = mesa_inputs.max_iterations,
            evaluations_filepath = mesa_inputs.evaluations_filepath,
            simulations_directory = mesa_inputs.simulations_directory,
        )


def logger_setup(settings_filepath):
    path = (
        settings_filepath[:-3]
        if settings_filepath.endswith(".py")
        else settings_filepath
    )
    logging.basicConfig(
        filename=path + ".log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Write to the screen as well
    logging.getLogger().addHandler(logging.StreamHandler())