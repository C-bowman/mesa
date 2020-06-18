
from numpy.random import random
from pandas import DataFrame, read_hdf

from os.path import isfile
from input_parsing import parse_inputs, check_dependencies, logger_setup
from sys import argv
import logging

from profile_models import linear_transport_profile, profile_radius_axis
from solps_interface import run_solps, evaluate_log_posterior, reset_solps
from parameter_sets import conductivity_profile, diffusivity_profile

def hypercube_sample(bounds):
    return [ b[0] + (b[1]-b[0])*random() for b in bounds ]

def uniform_sample(bounds):
    return bounds[0] + (bounds[1]-bounds[0])*random()


# check the validity of the input file and return its contents
settings = parse_inputs(argv)

# Check other data files are present
check_dependencies(settings, skip_training = True)

# set-up the log file
logger_setup(argv)

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

all_parameters = [key for key in fixed_parameter_values.keys()]
free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]


# first check if the training data file exists already, or needs to be created
if not isfile(output_directory + training_data_file):
    # define what columns will be in the dataframe
    cols = ['iteration',
            'gauss_logprob',
            'cauchy_logprob',
            'laplace_logprob',
            'prediction_mean',
            'prediction_error',
            'convergence_metric',
            *all_parameters]

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

    # create the dictionary for this iteration
    row_dict = {}

    # add values for all the fixed parameters
    for key in fixed_parameters:
        row_dict[key] = fixed_parameter_values[key]

    # sample values for all the free parameters
    for key in free_parameters:
        row_dict[key] = uniform_sample(optimisation_bounds[key])

    # produce transport profiles defined by new point
    radius = profile_radius_axis()

    chi_params = [row_dict[k] for k in conductivity_profile]
    chi = linear_transport_profile(radius, chi_params)

    D_params = [row_dict[k] for k in diffusivity_profile]
    D = linear_transport_profile(radius, D_params)
    dna = row_dict['D_div']
    hc  = row_dict['chi_div']
    
    logging.info('--- Starting iteration '+str(i)+' ---')
    logging.info('New chi parameters:')
    logging.info(chi_params)
    logging.info('New D parameters:')
    logging.info(D_params)
    logging.info('Divertor parameters:')
    logging.info([dna, hc])

    # Run SOLPS for the new point
    run_status = run_solps(chi=chi, chi_r=radius, D=D, D_r=radius, iteration = i, dna = dna, hci = hc, hce = hc,
                           run_directory = run_directory, output_directory = output_directory,
                           solps_n_timesteps = solps_n_timesteps, solps_dt = solps_dt, timeout_hours = solps_timeout_hours,
                           n_proc = solps_n_proc, n_species = solps_n_species, set_div_transport = set_divertor_transport)

    if run_status == False:
        logging.info('[initial_sampling] Restoring SOLPS run directory from reference.')
        reset_solps(run_directory,ref_directory)
        logging.info('[initial_sampling] Restoration complete, trying new run...')
        continue

    # evaluate the chi-squared
    logprobs = evaluate_log_posterior(iteration = i, directory = output_directory,
                                      diagnostic_data_files = diagnostic_data_files,
                                      diagnostic_data_observables = diagnostic_data_observables,
                                      diagnostic_data_errors = diagnostic_data_errors)

    gauss_logprob, cauchy_logprob, laplace_logprob = logprobs

    # build a new row for the dataframe
    row_dict['iteration'] = i
    row_dict['gauss_logprob'] = gauss_logprob
    row_dict['cauchy_logprob'] = cauchy_logprob
    row_dict['laplace_logprob'] = laplace_logprob
    row_dict['prediction_mean'] = None
    row_dict['prediction_error'] = None
    row_dict['convergence_metric'] = None

    df.loc[i] = row_dict # add the new row
    df.to_hdf(output_directory + training_data_file, key='training', mode='w') # save the data

    if i % solps_iter_reset == 0:
        reset_solps(run_directory,ref_directory)
