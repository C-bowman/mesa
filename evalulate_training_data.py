# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:34:30 2020

@author: James
"""


from mesa.solps import evaluate_log_posterior
from mesa.parsing import parse_inputs
import pandas as pd
import numpy as np

solps_dir = 'D:/solpsopt_runs/solpsopt_runs/tcv_58196_1280ms_init_2/'
orig_training_file = 'training_data_2.h5'
settings_file = 'settings.py'
output_file_name = 'training_data_new_2.h5'

settings = parse_inputs(['', solps_dir+settings_file])
output_file = solps_dir+output_file_name

# Read the optimiser settings
diagnostic_data_files = settings['diagnostic_data_files']
diagnostic_data_observables = settings['diagnostic_data_observables']
diagnostic_data_errors = settings['diagnostic_data_errors']

fixed_parameter_values = settings['fixed_parameter_values']

all_parameters = [key for key in fixed_parameter_values.keys()]
free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]
fixed_parameters = [key for key, value in fixed_parameter_values.items() if value is not None]

# Read the training data
orig_training_data = pd.read_hdf(solps_dir + orig_training_file)

# Create the new training file
cols = ['iteration',
        'gauss_logprob',
        'cauchy_logprob',
        'laplace_logprob',
        'prediction_mean',
        'prediction_error',
        'convergence_metric',
         *all_parameters]

# create the empty dataframe to store the training data and save it to HDF
df = pd.DataFrame(columns=cols)

df.to_hdf(output_file, key = 'training', mode = 'w')
del df

for itr in np.arange(len(orig_training_data)):

    # Extract the training data for this iteration
    run_data = orig_training_data.iloc[itr,:]
    
    iteration = run_data['iteration']
    conductivity_parameters = run_data['conductivity_parameters']
    diffusivity_parameters = run_data['diffusivity_parameters']
    div_parameters = run_data['div_parameters']
    
    
        
    df = pd.read_hdf(output_file, 'training')
    
    logprobs = evaluate_log_posterior(iteration = iteration, directory = solps_dir,
                                      diagnostic_data_files = diagnostic_data_files,
                                      diagnostic_data_observables = diagnostic_data_observables,
                                      diagnostic_data_errors = diagnostic_data_errors)
    
    gauss_logprob, cauchy_logprob, laplace_logprob = logprobs
    
    # create the dictionary for this iteration
    row_dict = {}
    
    # build a new row for the dataframe
    row_dict['chi_boundary_left'] = conductivity_parameters[0]
    row_dict['chi_boundary_right'] = conductivity_parameters[1]
    row_dict['chi_frac_left'] = conductivity_parameters[2]
    row_dict['chi_frac_right'] = conductivity_parameters[3]
    row_dict['chi_barrier_centre'] = conductivity_parameters[4]
    row_dict['chi_barrier_height'] = conductivity_parameters[5]
    row_dict['chi_barrier_width'] = conductivity_parameters[6]
    row_dict['chi_gap_left'] = conductivity_parameters[7]
    row_dict['chi_gap_right'] = conductivity_parameters[8]
    
    row_dict['D_boundary_left'] = diffusivity_parameters[0]
    row_dict['D_boundary_right'] = diffusivity_parameters[1]
    row_dict['D_frac_left'] = diffusivity_parameters[2]
    row_dict['D_frac_right'] = diffusivity_parameters[3]
    row_dict['D_barrier_centre'] = diffusivity_parameters[4]
    row_dict['D_barrier_height'] = diffusivity_parameters[5]
    row_dict['D_barrier_width'] = diffusivity_parameters[6]
    row_dict['D_gap_left'] = diffusivity_parameters[7]
    row_dict['D_gap_right'] = diffusivity_parameters[8]
    
    row_dict['D_div'] = div_parameters[0]
    row_dict['chi_div'] = div_parameters[1]
    
    row_dict['iteration'] = iteration
    row_dict['gauss_logprob'] = gauss_logprob
    row_dict['cauchy_logprob'] = cauchy_logprob
    row_dict['laplace_logprob'] = laplace_logprob
    row_dict['prediction_mean'] = None
    row_dict['prediction_error'] = None
    row_dict['convergence_metric'] = None
    
    if np.isnan(df.index.max()):
        df.loc[0] = row_dict
    else:
        df.loc[df.index.max()+1] = row_dict  # add the new row
    
    df.to_hdf(output_file, key='training', mode='w')  # save the data
    del df