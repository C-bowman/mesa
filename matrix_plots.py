
from runpy import run_path
from os.path import isfile
from sys import argv

from inference.plotting import matrix_plot
from pandas import read_hdf

# Get data from the settings module
if len(argv) == 1: # check to see if the settings module path was given
    raise ValueError('Path to settings module was not given as an argument')

if isfile(argv[1]): # check to see if the given path is valid
    # run the settings module
    settings = run_path(argv[1])
else:
    raise ValueError('{} is not a valid path to a settings module'.format(argv[1]))

# check that the settings module contains all the required information
keys = ['solps_output_directory', 'training_data_file', 'initial_sample_count']
for key in keys:
    if key not in settings:
        raise ValueError('"{}" was not found in the settings module'.format(key))

# extract the required information from settings
output_directory = settings['solps_output_directory']
training_data_file = settings['training_data_file']
initial_sample_count = settings['initial_sample_count']


# read and unpack the training data
df = read_hdf(output_directory + training_data_file, 'training')

start = initial_sample_count
stop = None
LP = df['log_posterior'][start:stop].to_numpy().copy()
X = df['conductivity_parameters'][start:stop]
D = df['diffusivity_parameters'][start:stop]

# normalise the probabilities to get colours for the points
cols = (LP-LP.min()) / (LP.max() - LP.min())

# build the samples for each parameter
X_samples = [[k[i] for k in X] for i in range(9)]
D_samples = [[k[i] for k in D] for i in range(9)]

# plot the Chi parameters
X_labels = [r'$\chi$ LHS height', r'$\chi$ RHS height', r'$\chi$ LHS frac', r'$\chi$ RHS frac',
            r'$\chi$ TB centre', r'$\chi$ TB height', r'$\chi$ TB width', r'$\chi$ LHS gap', r'$\chi$ RHS gap']
matrix_plot(X_samples, plot_style = 'scatter', point_colors = cols, colormap = 'viridis',
            point_size = 6, labels = X_labels, label_size = 9, filename = 'Chi_matrix_plot.pdf')

# plot the Diffusivity parameters
D_labels = [r'$D$ LHS height', r'$D$ RHS height', r'$D$ LHS frac', r'$D$ RHS frac',
            r'$D$ TB centre', r'$D$ TB height', r'$D$ TB width', r'$D$ LHS gap', r'$D$ RHS gap']

matrix_plot(D_samples, plot_style = 'scatter', point_colors = cols, colormap = 'viridis',
            point_size = 6, labels = D_labels, label_size = 9, filename = 'Diff_matrix_plot.pdf')