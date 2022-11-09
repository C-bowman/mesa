from runpy import run_path
from os.path import isfile
import logging

class Mesa:
    settings_filepath : str

    def __init__(self):
        self.settings = self.__parse_inputs(filepath)
        self.reffile = self.settings['training_data_file']
        self.settings_filepath = self.settings['ref_directory']
        self.driver = self.settings["driver"]
        self.simulation = self.settings["simulation"]
        self.optdata = {} # will become pandas dataframe of optimization iterations
    
    def run(self):
        # start the optimisation loop
        self.__init_datafile() # initialize data file of parameters
        # setup optimization (can include initial runs)
        self.driver.initialize(self.simulation)
        # run followup simulations (series of runs for opt/scan)
        while not self.driver.converged():
            # get the current iteration number
            itr = df['iteration'].max() + 1
            logging.info(f"--- Starting iteration {itr} ---")

            new_point = self.driver.get_next_point()

            # Run SOLPS for the new point
            Run = self.simulation.launch(
                iteration=itr,
                parameter_dictionary = new_point
            )

    def __init_datafile(self):
        # create the empty dataframe to store the training data and save it to HDF
        cols = self.driver.get_dataframe_columns()
        df = DataFrame(columns=cols)
        df.to_hdf(reference_directory + training_data_file, key='training', mode='w')
        del df

    def __parse_inputs(settings_filepath, check_training_data=False):
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
                [ MESA error ]
                >> The given string
                >> '{settings_filepath}'
                >> is not a valid path to a settings module.
                """
            )
        input_variables = [
            'simulation', # simulation object
            'driver', # optimizers, parameter scan, etc.
            'objective_function', # objective function to optimize
            'ref_directry', # directory where simulation data will go
            'training_data_file' # file for storing scan/opt data
        ]

        # check that the settings module contains all the required information
        for key in input_variables:
            if key not in settings:
                raise KeyError(
                    f"""
                    [ MESA error ]
                    >> The '{key}' variable was not
                    >> found in the settings file.
                    """
                )

        return settings

    def check_error_model(error_model):
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

    def logger_setup(settings_filepath):
        path = settings_filepath[:-3] if settings_filepath.endswith('.py') else settings_filepath
        logging.basicConfig(
            filename=path + '.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Write to the screen as well
        logging.getLogger().addHandler(logging.StreamHandler())