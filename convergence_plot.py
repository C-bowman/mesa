from numpy import maximum
import matplotlib.pyplot as plt
from pandas import read_hdf

from runpy import run_path
from os.path import isfile
from sys import argv

# Get data from the settings module
if len(argv) == 1: # check to see if the settings module path was given
    raise ValueError('Path to settings module was not given as an argument')

if isfile(argv[1]): # check to see if the given path is valid
    # run the settings module
    settings = run_path(argv[1])
else:
    raise ValueError('{} is not a valid path to a settings module'.format(argv[1]))

# check that the settings module contains all the required information
keys = ['solps_output_directory', 'optimisation_bounds', 'training_data_file', 'max_iterations']
for key in keys:
    if key not in settings:
        raise ValueError('"{}" was not found in the settings module'.format(key))

# extract the required information from settings
output_directory = settings['solps_output_directory']
optimisation_bounds = settings['optimisation_bounds']
training_data_file = settings['training_data_file']
max_iterations = settings['max_iterations']




df = read_hdf(output_directory + training_data_file, 'training')

itr = df['iteration']
LP = df['log_posterior']
running_max = maximum.accumulate(LP)

fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121)
ax1.plot(itr, running_max, '.-', c = 'red', label = 'highest observed value')
ax1.plot(itr, LP, 'o', c = 'C0', label = 'current value')
ax1.set_xlabel('iteration')
ax1.set_ylabel('posterior log-probability')
ax1.legend()
ax1.grid()

ax2 = fig.add_subplot(122)
ax2.plot(itr, df['expected_fractional_improvement'], 'o-')
ax2.set_xlabel('iteration')
ax2.set_ylabel('expected fractional improvement')
ax2.set_yscale('log')
ax2.grid()

plt.tight_layout()
plt.show()