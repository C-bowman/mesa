from numpy import maximum
import matplotlib.pyplot as plt
from pandas import read_hdf

from input_parsing import parse_inputs
from sys import argv

# check the validity of the input file and return its contents
settings = parse_inputs(argv)

# extract the required information from settings
output_directory = settings['solps_output_directory']
optimisation_bounds = settings['optimisation_bounds']
training_data_file = settings['training_data_file']
max_iterations = settings['max_iterations']
error_model = settings['error_model']




df = read_hdf(output_directory + training_data_file, 'training')

itr = df['iteration']
if error_model == 'Gaussian':
    LP = df['gauss_logprob'].to_numpy().copy()
elif error_model == 'Cauchy':
    LP = df['cauchy_logprob'].to_numpy().copy()

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
ax2.plot(itr, df['convergence_metric'], 'o-')
ax2.set_xlabel('iteration')
ax2.set_ylabel('convergence_metric')
ax2.set_yscale('log')
ax2.grid()

plt.tight_layout()
plt.show()
