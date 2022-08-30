from numpy import ndarray
from inference.gp import GpRegressor, GpOptimiser


def propose_evaluation(
    log_posterior: ndarray,
    normalised_parameters,
    hyperpar_bounds,
    settings: dict
):
    # construct the GP
    covariance_kernel = settings["covariance_kernel"](hyperpar_bounds=hyperpar_bounds)
    GP = GpRegressor(
        normalised_parameters,
        log_posterior,
        cross_val=settings["cross_validation"],
        kernel=covariance_kernel,
        optimizer="diffev"
    )
    bfgs_hps = GP.multistart_bfgs(starts=300, n_processes=settings["solps_n_proc"])

    if GP.model_selector(bfgs_hps) > GP.model_selector(GP.hyperpars):
        optimal_hyperpars = bfgs_hps
    else:
        optimal_hyperpars = GP.hyperpars

    # If a trust-region approach is being used, limit the search area
    # to a region around the current maximum
    trhw = 0.5 * settings["trust_region_width"]
    if settings["trust_region"]:
        max_ind = log_posterior.argmax()
        max_point = normalised_parameters[max_ind]
        search_bounds = [(max(0., v - trhw), min(1., v + trhw)) for v in max_point]
    else:
        search_bounds = [(0., 1.) for _ in range(normalised_parameters[0].size)]

    # build the GP-optimiser
    covariance_kernel = settings["covariance_kernel"](hyperpar_bounds=hyperpar_bounds)
    GPopt = GpOptimiser(
        normalised_parameters,
        log_posterior,
        hyperpars=optimal_hyperpars,
        bounds=search_bounds,
        cross_val=settings["cross_validation"],
        kernel=covariance_kernel,
        acquisition=settings['acquisition_function']
    )

    # maximise the acquisition both by multi-start bfgs and differential evolution,
    # and use the best of the two
    bfgs_prop = GPopt.propose_evaluation(optimizer="bfgs")
    diff_prop = GPopt.propose_evaluation(optimizer="diffev")

    bfgs_acq = GPopt.acquisition(bfgs_prop)
    diff_acq = GPopt.acquisition(diff_prop)

    new_point = bfgs_prop if bfgs_acq > diff_acq else diff_prop

    # calculate the convergence metric
    convergence = GPopt.acquisition.convergence_metric(new_point)

    # get predicted log-probability at the new point
    mu_lp, sigma_lp = GPopt.gp(new_point)

    metrics = {
        "prediction_mean": mu_lp,
        "prediction_error": sigma_lp,
        "prediction_metric": convergence
    }

    return new_point, metrics
