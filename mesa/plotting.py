
from numpy import array, concatenate, maximum, mean
import matplotlib.pyplot as plt
from pandas import read_hdf

from mesa import bounds_transform
from mesa.parsing import parse_inputs, check_error_model
from mesa.parameters import conductivity_profile, diffusivity_profile
from inference.plotting import matrix_plot
from inference.gp import GpRegressor


def convergence_plot(settings_filepath):
    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # extract the required information from settings
    output_directory = settings['solps_output_directory']
    optimisation_bounds = settings['optimisation_bounds']
    training_data_file = settings['training_data_file']
    max_iterations = settings['max_iterations']
    error_model = settings['error_model']

    df = read_hdf(output_directory + training_data_file, 'training')

    itr = df['iteration']
    logprob_key = check_error_model(error_model)
    LP = df[logprob_key].to_numpy().copy()

    running_max = maximum.accumulate(LP)

    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121)
    ax1.plot(itr, running_max, '.-', c='red', label='highest observed value')
    ax1.plot(itr, LP, 'o', c='C0', label='current value')
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


def matrix_plots(settings_filepath):
    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath)

    # extract the required information from settings
    output_directory = settings['solps_output_directory']
    training_data_file = settings['training_data_file']
    initial_sample_count = settings['initial_sample_count']
    error_model = settings['error_model']

    # read and unpack the training data
    df = read_hdf(output_directory + training_data_file, 'training')

    start = initial_sample_count
    stop = None

    logprob_key = check_error_model(error_model)
    LP = df[logprob_key][start:stop].to_numpy().copy()

    X = [df[key][start:stop] for key in conductivity_profile]
    D = [df[key][start:stop] for key in diffusivity_profile]

    # normalise the probabilities to get colours for the points
    cols = (LP-LP.min()) / (LP.max() - LP.min())

    # build the samples for each parameter
    X_samples = [[k[i] for k in X] for i in range(9)]
    D_samples = [[k[i] for k in D] for i in range(9)]

    # plot the Chi parameters
    X_labels = [
        r'$\chi$ LHS height', r'$\chi$ RHS height', r'$\chi$ LHS frac',
        r'$\chi$ RHS frac', r'$\chi$ TB centre', r'$\chi$ TB height',
        r'$\chi$ TB width', r'$\chi$ LHS gap', r'$\chi$ RHS gap'
    ]
    matrix_plot(
        X_samples, plot_style='scatter', point_colors=cols, colormap='viridis',
        point_size=6, labels=X_labels, label_size=9, filename='Chi_matrix_plot.pdf'
    )

    # plot the Diffusivity parameters
    D_labels = [
        r'$D$ LHS height', r'$D$ RHS height', r'$D$ LHS frac',
        r'$D$ RHS frac', r'$D$ TB centre', r'$D$ TB height',
        r'$D$ TB width', r'$D$ LHS gap', r'$D$ RHS gap'
    ]
    matrix_plot(
        D_samples, plot_style='scatter', point_colors=cols, colormap='viridis',
        point_size=6, labels=D_labels, label_size=9, filename='Diff_matrix_plot.pdf'
    )


def cross_validation_plot(settings_filepath):
    # check the validity of the input file and return its contents
    settings = parse_inputs(settings_filepath, check_training_data=True)

    # extract the required information from settings
    ref_directory = settings['solps_ref_directory']
    optimisation_bounds = settings['optimisation_bounds']
    training_data_file = settings['training_data_file']
    cross_validation = settings['cross_validation']
    error_model = settings['error_model']
    covariance_kernel = settings['covariance_kernel']
    fixed_parameter_values = settings['fixed_parameter_values']
    initial_sample_count = settings['initial_sample_count']

    free_parameters = [key for key, value in fixed_parameter_values.items() if value is None]

    # load the training data
    df = read_hdf(ref_directory + training_data_file)
    logprob_key = check_error_model(error_model)
    log_posterior = df[logprob_key].to_numpy().copy()

    parameters = [df[p] for p in free_parameters]

    # convert the data to the normalised coordinates:
    training_points = [bounds_transform(p, optimisation_bounds) for p in parameters]

    # if requested, normalise the training data to have zero mean
    data_mean = mean(log_posterior[:initial_sample_count])
    log_posterior -= data_mean

    # construct the GP
    GP = GpRegressor(
        training_points,
        log_posterior,
        cross_val=cross_validation,
        kernel=covariance_kernel,
        optimizer="diffev"
    )
    bfgs_hps = GP.multistart_bfgs(starts=300)

    if GP.model_selector(bfgs_hps) > GP.model_selector(GP.hyperpars):
        GP.set_hyperparameters(bfgs_hps)

    # get the LOO predictions
    mu_loo, sigma_loo = GP.loo_predictions()

    # build the cross-validation plot
    import matplotlib.pyplot as plt
    upr = max(log_posterior.max(), mu_loo.max())
    lwr = min(log_posterior.min(), mu_loo.min())
    upr += (upr-lwr)*0.1
    lwr -= (upr-lwr)*0.1

    plt.errorbar(log_posterior, mu_loo, yerr=sigma_loo, ls='none', marker='.')
    plt.plot([lwr, upr], [lwr, upr], ls='dashed', c='black')
    plt.xlim([lwr, upr])
    plt.ylim([lwr, upr])
    plt.ylabel('GP prediction of left-out point')
    plt.xlabel('value of left-out point')
    plt.grid()
    plt.tight_layout()
    plt.show()
