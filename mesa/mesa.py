import logging
from pathlib import Path
from pandas import DataFrame
from mesa.simulations import Simulation
from mesa.strategies import Strategy
from mesa.input import parse_inputs


class Mesa:
    def __init__(self, filepath: Path | str):
        mesa_inputs = parse_inputs(filepath)
        self.evaluations_filepath = mesa_inputs.evaluations_filepath
        self.simulations_directory = mesa_inputs.simulations_directory
        self.strategy: Strategy = mesa_inputs.strategy
        self.simulation: Simulation = mesa_inputs.simulation
        self.objective_function = mesa_inputs.objective_function
        self.optdata = {}  # will become pandas dataframe of optimization iterations

    def run(self):
        # start the optimisation loop
        self.__init_datafile()  # initialize data file of parameters
        # setup optimization (can include initial runs)
        self.strategy.initialize(
            simulation=self.simulation,
            objective_func=self.objective_function,
            training_file=self.simulations_directory / self.evaluations_filepath,
        )
        # run followup simulations (series of runs for opt/scan)
        self.strategy.run()

    def __init_datafile(self):
        # create the empty dataframe to store the training data and save it to HDF
        cols = self.strategy.get_dataframe_columns()
        df = DataFrame(columns=cols)
        df.to_hdf(
            self.simulations_directory / self.evaluations_filepath,
            key="training",
            mode="w",
        )
        del df

    def logger_setup(self, settings_filepath):
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
