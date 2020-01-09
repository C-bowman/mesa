from numpy import array, concatenate, mean
from pandas import read_hdf

from input_parsing import parse_inputs
from sys import argv

from inference.gp_tools import GpRegressor
from inference.mcmc import GibbsChain, ParallelTempering

def bounds_transform(v, bounds, inverse=False):
    if inverse:
        return array([b[0] + (b[1]-b[0])*k for k,b in zip(v, bounds)])
    else:
        return array([(k-b[0])/(b[1]-b[0]) for k, b in zip(v, bounds)])


# check the validity of the input file and return its contents
settings = parse_inputs(argv)

# extract the required information from settings
output_directory = settings['solps_output_directory']
optimisation_bounds = settings['optimisation_bounds']
training_data_file = settings['training_data_file']
diagnostic_data_file = settings['diagnostic_data_file']
max_iterations = settings['max_iterations']
cross_validation = settings['cross_validation']
covariance_kernel = settings['covariance_kernel']
fixed_parameter_values = settings['fixed_parameter_values']
initial_sample_count = settings['initial_sample_count']


# build the indices for the varied vs fixed parameters:
varied_inds  = array([ i for i,v in enumerate(fixed_parameter_values) if v is None])
fixed_inds   = array([ i for i,v in enumerate(fixed_parameter_values) if v is not None])
fixed_values = array([ v for i,v in enumerate(fixed_parameter_values) if v is not None])


# load the training data
df = read_hdf(output_directory+training_data_file)

# extract the training data
log_posterior = df['log_posterior'].to_numpy().copy()
parameters = []
for X, D, H in zip(df['conductivity_parameters'], df['diffusivity_parameters'], df['div_parameters']):
    parameters.append(concatenate([X, D, H]))

# convert the data to the normalised coordinates:
normalised_parameters = [bounds_transform(p,optimisation_bounds) for p in parameters]

# get the training points by extracting the parameters which are to be optimised
# from the normalised parameter vectors:
training_points = [v[varied_inds] for v in normalised_parameters]

# if requested, normalise the training data to have zero mean
data_mean = mean(log_posterior[:initial_sample_count])
log_posterior -= data_mean

# construct the GP
GP = GpRegressor(training_points, log_posterior, cross_val = cross_validation, kernel = covariance_kernel)

# define a set of temperature levels
N_levels = 6
temps = [10**(2.*k/(N_levels-1.)) for k in range(N_levels)]

# create a set of chains - one with each temperature
bounds = GP.cov.get_bounds()
chains = [ GibbsChain(posterior=GP.model_selector, start = [0.5*(b[0] + b[1]) for b in bounds], temperature=T) for T in temps ]
for chain in chains:
    for i, b in enumerate(bounds): chain.set_boundaries(i, b)

# When an instance of ParallelTempering is created, a dedicated process for each chain is spawned.
# These separate processes will automatically make use of the available cpu cores, such that the
# computations to advance the separate chains are performed in parallel.
PT = ParallelTempering(chains=chains)

PT.run_for(minutes=2)

chains = PT.return_chains()
PT.shutdown()

# PT.swap_diagnostics()

# extract the hyper-parameter estimate
mode = chains[0].mode()

# if MCMC found a better solution then use it
de_score = GP.model_selector(GP.hyperpars)
mc_score = GP.model_selector(mode)

if mc_score > de_score:
    GP.set_hyperparameters(mode)

print("""
-------------------------
| DiffEv score: {}
|   MCMC score: {}
-------------------------
""".format(de_score/len(training_points), mc_score/len(training_points))
      )

# get the LOO predictions
mu_loo, sigma_loo = GP.loo_predictions()

# build the cross-validation plot
import matplotlib.pyplot as plt
upr = max(log_posterior.max(), mu_loo.max())
lwr = min(log_posterior.min(), mu_loo.min())
upr += (upr-lwr)*0.1
lwr -= (upr-lwr)*0.1

plt.errorbar(log_posterior, mu_loo, yerr=sigma_loo, ls = 'none', marker = '.')
plt.plot([lwr,upr], [lwr,upr], ls = 'dashed', c = 'black')
plt.xlim([lwr,upr])
plt.ylim([lwr,upr])
plt.ylabel('GP prediction of left-out point')
plt.xlabel('value of left-out point')
plt.grid()
plt.tight_layout()
plt.show()