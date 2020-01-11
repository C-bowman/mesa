
from numpy import array
from numpy.random import random
from pandas import DataFrame, read_hdf

from os.path import isfile
from input_parsing import parse_inputs, check_dependencies
from sys import argv

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior, reset_solps

def hypercube_sample(bounds):
    return [ b[0] + (b[1]-b[0])*random() for b in bounds ]

# check the validity of the input file and return its contents
settings = parse_inputs(argv)

# Check other data files are present
check_dependencies(settings, skip_training = True)

# data & results filepaths
run_directory = settings['solps_run_directory']
output_directory = settings['solps_output_directory']
ref_directory = settings['solps_ref_directory']
training_data_file = settings['training_data_file']
diagnostic_data_files = settings['diagnostic_data_files']
diagnostic_data_observables = settings['diagnostic_data_observables']
diagnostic_data_errors = settings['diagnostic_data_errors']

# SOLPS settings
solps_n_timesteps = settings['solps_n_timesteps']
solps_dt = settings['solps_dt']
solps_n_proc = settings['solps_n_proc']
solps_iter_reset = settings['solps_iter_reset']
solps_n_species = settings['solps_n_species']
solps_timeout_hours = settings['solps_timeout_hours']
set_divertor_transport = settings['set_divertor_transport']

# optimiser settings
fixed_parameter_values = settings['fixed_parameter_values']
optimisation_bounds = settings['optimisation_bounds']
initial_sample_count = settings['initial_sample_count']





# build the indices for the varied vs fixed parameters:
varied_inds  = array([ i for i,v in enumerate(fixed_parameter_values) if v is None])
fixed_inds   = array([ i for i,v in enumerate(fixed_parameter_values) if v is not None])
fixed_values = array([ v for i,v in enumerate(fixed_parameter_values) if v is not None])


# first check if the training data file exists already, or needs to be created
if not isfile(output_directory + training_data_file):
    # define what columns will be in the dataframe
    cols = ['iteration',
            'conductivity_parameters',
            'diffusivity_parameters',
            'div_parameters',
            'log_posterior',
            'prediction_mean',
            'prediction_error',
            'convergence_metric' ]

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
    new_parameters = array(hypercube_sample(optimisation_bounds))

    # if any of the parameters have been fixed, then insert those values
    if len(fixed_values) > 0:
        new_parameters[fixed_inds] = fixed_values

    # produce transport profiles defined by new point
    radius = profile_radius_axis()

    chi = linear_transport_profile(radius, new_parameters[0:9])
    D = linear_transport_profile(radius, new_parameters[9:18])
    dna = new_parameters[18]
    hc  = new_parameters[19]

    # Run SOLPS for the new point
    run_status = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, dna = dna, hci = hc, hce = hc,
                           run_directory = run_directory, output_directory = output_directory,
                           solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt, timeout_hours = solps_timeout_hours,
                           n_proc = solps_n_proc, n_species = solps_n_species, set_div_transport = set_divertor_transport)

    if run_status == False:
        print('[initial_sampling] Restoring SOLPS run directory from reference.')
        reset_solps(run_directory,ref_directory)
        print('[initial_sampling] Restoration complete, trying new run...')
        continue

    # evaluate the chi-squared
    log_posterior = evaluate_log_posterior(iteration = i, directory = output_directory,
                                           diagnostic_data_files = diagnostic_data_files,
                                           diagnostic_data_observables = diagnostic_data_observables,
                                           diagnostic_data_errors = diagnostic_data_errors)

    # build a new row for the dataframe
    row_dict = {
        'iteration' : i,
        'conductivity_parameters' : new_parameters[0:9],
        'diffusivity_parameters' : new_parameters[9:18],
        'div_parameters' : new_parameters[18:20],
        'log_posterior' : log_posterior,
        'prediction_mean' : None,
        'prediction_error' : None,
        'convergence_metric' : None
    }

    df.loc[i] = row_dict # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w') # save the data

    if i % solps_iter_reset == 0:
        reset_solps(run_directory,ref_directory)