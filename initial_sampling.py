
from numpy.random import random
from pandas import DataFrame, read_hdf

from runpy import run_path
from os.path import isfile
from sys import argv

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior

def hypercube_sample(bounds):
    return [ b[0] + (b[1]-b[0])*random() for b in bounds ]



# Get data from the settings module
if len(argv) == 1: # check to see if the settings module path was given
    raise ValueError('Path to settings module was not given as an argument')

if isfile(argv[1]): # check to see if the given path is valid
    # run the settings module
    settings = run_path(argv[1])
else:
    raise ValueError('{} is not a valid path to a settings module'.format(argv[1]))

# check that the settings module contains all the required information
keys = ['solps_run_directory', 'solps_output_directory', 'optimisation_bounds', 'training_data_file', 'diagnostic_data_file', 'initial_sample_count','solps_n_timesteps','solps_dt']
for key in keys:
    if key not in settings:
        raise ValueError('"{}" was not found in the settings module'.format(key))

# extract the required information from settings
run_directory = settings['solps_run_directory']
output_directory = settings['solps_output_directory']
optimisation_bounds = settings['optimisation_bounds']
training_data_file = settings['training_data_file']
diagnostic_data_file = settings['diagnostic_data_file']
diagnostic_data_desc = settings['diagnostic_data_desc']
initial_sample_count = settings['initial_sample_count']
solps_n_timesteps = settings['solps_n_timesteps']
solps_dt = settings['solps_dt']
solps_n_proc = settings['solps_n_proc']

# first check if the training data file exists already, or needs to be created
if not isfile(output_directory + training_data_file):
    # define what columns will be in the dataframe
    cols = ['iteration',
            'conductivity_parameters',
            'diffusivity_parameters',
            'log_posterior',
            'prediction_mean',
            'prediction_error',
            'expected_fractional_improvement' ]

    # create the empty dataframe to store the training data and save it to HDF
    df = DataFrame(columns=cols)

    df.to_hdf(output_directory + training_data_file, key = 'training', mode = 'w')
    del df



# loop until enough samples have been evaluated
while True:
    df = read_hdf(output_directory + training_data_file, 'training')

    # get the current iteration number
    if df['iteration'].size == 0:
        i = 1
    else:
        i = df['iteration'].max() + 1

    # break the loop if we've hit the desired number of initial samples
    if i > initial_sample_count: break

    # sample new evaluation point
    new_point = hypercube_sample(optimisation_bounds)

    # produce transport profiles defined by new point
    radius = profile_radius_axis()
    
    L = len(new_point) // 2
    chi = linear_transport_profile(radius, new_point[:L])
    D = linear_transport_profile(radius, new_point[L:])

    # Run SOLPS for the new point
    run_id = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, run_directory = run_directory,
                       output_directory = output_directory, solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt,
                       n_proc = solps_n_proc)

    # evaluate the chi-squared
    log_posterior = evaluate_log_posterior(iteration = i, directory = output_directory,
                                           diagnostic_data_file = diagnostic_data_file,
                                           diagnostic_data_desc = diagnostic_data_desc)

    # build a new row for the dataframe
    row_dict = {
        'iteration' : i,
        'conductivity_parameters' : new_point[:L],
        'diffusivity_parameters' : new_point[L:],
        'log_posterior' : log_posterior,
        'prediction_mean' : None,
        'prediction_error' : None,
        'expected_fractional_improvement' : None
    }

    df.loc[i] = row_dict # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w') # save the data
