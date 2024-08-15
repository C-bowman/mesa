from runpy import run_path
from os.path import isfile
import logging
from pandas import DataFrame
from mesa.simulations import Simulation
from mesa.strategies import Strategy


class Mesa:
    settings_filepath: str

    def __init__(self, filepath=None):
        self.settings = self.parse_inputs(filepath)
        self.training_data_file = self.settings["training_data_file"]
        self.reference_directory = self.settings["ref_directory"]
        self.strategy: Strategy = self.settings["strategy"]
        self.simulation: Simulation = self.settings["simulation"]
        self.objective_function = self.settings["objective_function"]
        self.optdata = {}  # will become pandas dataframe of optimization iterations

    def run(self):
        # start the optimisation loop
        self.__init_datafile()  # initialize data file of parameters
        # setup optimization (can include initial runs)
        self.strategy.initialize(
            simulation=self.simulation,
            objective_func=self.objective_function,
            training_file=self.reference_directory + "/" + self.training_data_file
        )
        # run followup simulations (series of runs for opt/scan)
        self.strategy.run()

    def __init_datafile(self):
        # create the empty dataframe to store the training data and save it to HDF
        cols = self.strategy.get_dataframe_columns()
        df = DataFrame(columns=cols)
        df.to_hdf(self.reference_directory + self.training_data_file, key="training", mode="w")
        del df

    def parse_inputs(self, settings_filepath, check_training_data=False):
        """
        Checks whether the settings file exists, and contains all necessary
        fields, then returns its contents as a dictionary.

        :param settings_filepath: The path to the settings file.
        :return: Dictionary containing the settings file data.
        """
        if type(settings_filepath) is not str:
            raise TypeError(
                f"""
                [ MESA error ]
                >> Settings file path must be a string.
                >> Instead type {type(settings_filepath)} was given.
                """
            )

        if isfile(settings_filepath):  # check to see if the given path is valid
            settings = run_path(settings_filepath)  # run the settings module
        else:
            raise FileNotFoundError(
                f"""
                [ MESA error ] \
                >> The given string \
                >> '{settings_filepath}' \
                >> is not a valid path to a settings module.
                """
            )
        input_variables = [
            "simulation",  # simulation object
            "driver",  # optimizers, parameter scan, etc.
            "objective_function",  # objective function to optimize
            "ref_directory",  # directory where simulation data will go
            "training_data_file",  # file for storing scan/opt data
        ]

        # check that the settings module contains all the required information
        for key in input_variables:
            if key not in settings:
                raise KeyError(
                    f"""
                    [ MESA error ] \
                    >> The '{key}' variable was not \
                    >> found in the settings file.
                    """
                )

        return settings

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
