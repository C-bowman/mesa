
from runpy import run_path
from os.path import isfile

input_variables = [
# directory settings
    'solps_run_directory',
    'solps_ref_directory',
    'solps_output_directory',
    'training_data_file',
    'diagnostic_data_files',
    'diagnostic_data_observables',
    'diagnostic_data_errors',
# SOLPS settings
    'solps_n_timesteps',
    'solps_dt',
    'solps_n_proc',
    'solps_iter_reset',
    'set_divertor_transport',
    'solps_n_species',
    'solps_timeout_hours',
# gaussian-process regression settings
    'cross_validation',
    'covariance_kernel',
# optimiser settings
    'initial_sample_count',
    'max_iterations',
    'acquisition_function',
    'trust_region',
    'trust_region_width',
    'optimisation_bounds',
    'fixed_parameter_values'
]


def parse_inputs(args):
    # Get data from the settings module
    if len(args) == 1:  # check to see if the settings module path was given
        raise ValueError('Path to settings module was not given as an argument')

    if isfile(args[1]):  # check to see if the given path is valid
        settings = run_path(args[1])  # run the settings module
    else:
        raise ValueError('{} is not a valid path to a settings module'.format(args[1]))

    # check that the settings module contains all the required information
    for key in input_variables:
        if key not in settings:
            raise ValueError('"{}" was not found in the settings module'.format(key))
    return settings


def check_dependencies(settings):
    """
    Checks that the files required to run the optimiser are present.
    If a file is missing, an exception is raised.
    """
    output_directory = settings['solps_output_directory']
    training_data_file = settings['training_data_file']
    diagnostic_data_files = settings['diagnostic_data_files']

    # Check if diagnostic data is present
    for df in diagnostic_data_files:
        if not isfile(output_directory+df+'.h5'):
            raise Exception('File not found: '+output_directory+df+'.h5')

    # Check if training data is present
    if not isfile(output_directory+training_data_file):
        raise Exception('File not found: '+output_directory+training_data_file)