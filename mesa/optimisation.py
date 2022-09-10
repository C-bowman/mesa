from numpy import ndarray
from inference.gp import GpRegressor, GpOptimiser


def propose_evaluation(
    log_posterior: ndarray,
    normalised_parameters,
    kernel,
    mean_function,
    acquisition,
    cross_validation: bool,
    n_procs: int,
    trust_region_width=None,
):
    GP = GpRegressor(
        normalised_parameters,
        log_posterior,
        cross_val=cross_validation,
        kernel=kernel,
        mean=mean_function,
        optimizer="bfgs",
        n_processes=n_procs,
        n_starts=300
    )

    # If a trust-region approach is being used, limit the search area
    # to a region around the current maximum
    if trust_region_width is not None:
        trhw = 0.5 * trust_region_width
        max_ind = log_posterior.argmax()
        max_point = normalised_parameters[max_ind]
        search_bounds = [(max(0., v - trhw), min(1., v + trhw)) for v in max_point]
    else:
        search_bounds = [(0., 1.) for _ in range(normalised_parameters[0].size)]

    # build the GP-optimiser
    GPopt = GpOptimiser(
        normalised_parameters,
        log_posterior,
        hyperpars=GP.hyperpars,
        bounds=search_bounds,
        cross_val=cross_validation,
        kernel=kernel,
        mean=mean_function,
        acquisition=acquisition,
        n_processes=n_procs
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
