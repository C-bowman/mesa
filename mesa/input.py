from runpy import run_path
from dataclasses import dataclass
from pathlib import Path
from os.path import isfile
from inspect import get_annotations

from mesa.simulations import Simulation
from mesa.strategies import Strategy
from mesa.diagnostics import ObjectiveFunction


@dataclass
class MesaInputs:
    simulation: Simulation
    strategy: Strategy
    objective_function: ObjectiveFunction
    evaluations_filepath: Path
    simulations_directory: Path

    def __post_init__(self):
        assert self.evaluations_filepath.exists()
        assert self.evaluations_filepath.is_file()
        assert self.simulations_directory.exists()
        assert self.simulations_directory.is_dir()


def parse_inputs(
    settings_filepath: Path | str
) -> MesaInputs:
    """
    Checks whether the settings file exists, and contains all necessary
    fields, then returns its contents as a dictionary.

    :param settings_filepath: The path to the settings file.
    :return: Dictionary containing the settings file data.
    """
    if not isinstance(settings_filepath, (Path, str)):
        raise TypeError(
            f"""\n
            \r[ MESA error ]
            \r>> Settings file path must be a string.
            \r>> Instead type {type(settings_filepath)} was given.
            """
        )

    if isfile(settings_filepath):  # check to see if the given path is valid
        settings = run_path(settings_filepath)  # run the settings module
    else:
        raise FileNotFoundError(
            f"""\n
            \r[ MESA error ]
            \r>> The given string
            \r>> '{settings_filepath}'
            \r>> is not a valid path to a settings module.
            """
        )

    annotations = get_annotations(MesaInputs)

    # verify all the required variables exist in the input file
    for variable_name in annotations.keys():
        if variable_name not in settings:
            raise KeyError(
                f"""\n
                \r[ MESA error ]
                \r>> The '{variable_name}' variable was not
                \r>> found in the settings file.
                """
            )

    # verify all variables have the requested types
    for variable_name, requested_type in annotations.items():
        if not isinstance(settings[variable_name], requested_type):
            raise TypeError(
                f"""\n
                \r[ MESA error ]
                \r>> The '{variable_name}' variable should be an instance of:
                \r>> {requested_type}
                \r>> but instead has type:
                \r>> {type(settings[variable_name])}
                """
            )

    parsed_inputs = {
        variable_name: settings[variable_name]
        for variable_name in annotations.keys()
    }

    return MesaInputs(**parsed_inputs)
