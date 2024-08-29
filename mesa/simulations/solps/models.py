from collections.abc import Sequence
from numpy import array, ndarray, piecewise, concatenate, sort, zeros


def middle_section(x: ndarray, h: float) -> float:
    return h


def linear_section(x: ndarray, m: float, c: float) -> ndarray:
    return m * x + c


def linear_profile_knots(
    params: Sequence[float], boundaries: tuple[float, float]
) -> tuple[ndarray, ndarray]:
    knot_locations = array(
        [
            boundaries[0],  # left edge
            boundaries[1],  # right edge
            params[4] - 0.5 * params[6],
            params[4] + 0.5 * params[6],
            params[4] - 0.5 * params[6] - params[7],
            params[4] + 0.5 * params[6] + params[8],
        ]
    )

    knot_values = array(
        [
            params[0] + params[5],
            params[1] + params[5],
            params[5],
            params[5],
            params[5] + params[2] * params[0],
            params[5] + params[3] * params[1],
        ]
    )
    sorter = knot_locations.argsort()
    return knot_locations[sorter], knot_values[sorter]


def linear_transport_profile(
    x: ndarray, params: Sequence[float], boundaries=(-0.1, 0.1)
) -> ndarray:
    M_height = params[5]  # height of the middle section
    L_asymp = params[0] + M_height  # left boundary height
    R_asymp = params[1] + M_height  # right asymptote height
    L_mid = params[2] * params[0] + M_height  # left-middle height
    R_mid = params[3] * params[1] + M_height  # left-middle height
    L_bound = params[4] - 0.5 * params[6]  # boundary between left and middle sections
    R_bound = params[4] + 0.5 * params[6]  # boundary between middle and right sections
    L_gap = params[7]  # left-mid-point gap from LM boundary
    R_gap = params[8]  # right-mid-point gap from MR boundary

    # construct boolean arrays of the conditions
    in_left = x < L_bound - L_gap
    in_right = x > R_bound + R_gap
    in_left_mid = (x < L_bound) & (x >= L_bound - L_gap)
    in_right_mid = (x > R_bound) & (x <= R_bound + R_gap)
    in_middle = ~(in_left | in_right | in_left_mid | in_right_mid)
    conditions = [in_left, in_left_mid, in_middle, in_right_mid, in_right]

    # get the line gradients for each section
    Lx, Rx = boundaries
    L_m = (L_mid - L_asymp) / (L_bound - L_gap - Lx)
    R_m = (R_asymp - R_mid) / (Rx - R_bound - R_gap)
    ML_m = (M_height - L_mid) / L_gap
    MR_m = (R_mid - M_height) / R_gap

    # get the line y-intercepts for each section
    L_c = L_asymp - L_m * Lx
    R_c = R_asymp - R_m * Rx
    ML_c = L_mid - ML_m * (L_bound - L_gap)
    MR_c = R_mid - MR_m * (R_bound + R_gap)

    # build functions for each section
    functions = [
        lambda z: linear_section(z, L_m, L_c),
        lambda z: linear_section(z, ML_m, ML_c),
        lambda z: middle_section(z, M_height),
        lambda z: linear_section(z, MR_m, MR_c),
        lambda z: linear_section(z, R_m, R_c),
    ]

    return piecewise(x, conditions, functions)


def profile_radius_axis(
    params: Sequence[float], boundaries: tuple[float, float]
) -> ndarray:
    r, _ = linear_profile_knots(params, boundaries)
    # find the spacing that we'll use to place points around each knot
    dr = [r[1] - r[0]]
    for i in range(1, 5):
        dr.append(min(r[i] - r[i - 1], r[i + 1] - r[i]))
    dr.append(r[5] - r[4])
    # generate the points around each knot
    spacing = array([-6.0, -2.0, -1.0, 1.0, 2.0, 6.0]) * 0.03
    left_edge = spacing[3:] * dr[0] + r[0]
    right_edge = spacing[:3] * dr[-1] + r[-1]
    middles = [spacing * dr[i] + r[i] for i in range(1, 5)]
    # combine all the points and return them sorted
    return sort(concatenate([left_edge, *middles, right_edge, boundaries]))


def triangle_cdf(x: ndarray, start: float, end: float) -> ndarray:
    """
    Returns the cumulative distribution function for the (symmetric) triangle
    distribution which spans the interval [start, end].
    """
    mid = 0.5 * (start + end)
    y = zeros(x.size)

    left = (x > start) & (x <= mid)
    right = (x > mid) & (x < end)

    y[left] = (x[left] - start) ** 2 / ((end - start) * (mid - start))
    y[right] = 1 - (x[right] - end) ** 2 / ((end - start) * (end - mid))
    y[x >= end] = 1.0
    return y


def smooth_ramp(
    x: ndarray, start: float, end: float, gradient: float, right_side=True
) -> tuple[ndarray, float]:
    y = zeros(x.size)
    dx = end - start
    inside = (x > start) & (x <= end)
    if right_side:
        after = x > end
        y[inside] = 0.5 * gradient * (x[inside] - start)**2 / dx
        y[after] = gradient * (x[after] + (0.5 * dx - end))
        boundary_val = gradient * 0.5 * dx
    else:
        after = x < start
        y[inside] = -0.5 * gradient * (end - x[inside])**2 / dx
        y[after] = gradient * (x[after] + (0.5 * dx - end))
        boundary_val = -gradient * 0.5 * dx
    return y, boundary_val


def smooth_barrier_edge(
    x: ndarray, start: float, end: float, gradient: float, right_side=True
) -> ndarray:
    ramp, bound_value = smooth_ramp(x, start, end, gradient, right_side=right_side)
    cdf = triangle_cdf(x, start, end)
    sigmoid = cdf if right_side else 1 - cdf
    return ramp + sigmoid * (1 - bound_value)


def smooth_transport_profile(x: ndarray, params: Sequence[float]) -> ndarray:
    """
    A smooth transport profile model with a continuous first-derivative where
    the flat sections of the core, transport barrier and SOL are connected using
    the CDF of a triangle distribution.
    """
    y_core, y_tb, y_sol, x_tb, w_tb, core_rise, sol_rise, core_grad, sol_grad = params
    left_end = x_tb - 0.5 * w_tb
    left_start = left_end - core_rise
    left_barrier = smooth_barrier_edge(
        x, left_start, left_end, core_grad, right_side=False
    )

    right_start = x_tb + 0.5 * w_tb
    right_end = right_start + sol_rise
    right_barrier = smooth_barrier_edge(
        x, right_start, right_end, sol_grad, right_side=True
    )
    return left_barrier * (y_core - y_tb) + right_barrier * (y_sol - y_tb) + y_tb


def smooth_profile_knots(
    params: Sequence[float], boundaries: tuple[float, float]
) -> ndarray:
    y_core, y_tb, y_sol, x_tb, w_tb, core_rise, sol_rise = params
    knots = [
        boundaries[0],
        x_tb - 0.5 * w_tb - core_rise,
        x_tb - 0.5 * w_tb,
        x_tb + 0.5 * w_tb,
        x_tb + 0.5 * w_tb + sol_rise,
        boundaries[1],
    ]

    shifts = [1 / 6, 2 / 6, 5 / 12]
    shifts.extend([-s for s in shifts])

    knots.extend([x_tb - 0.5 * w_tb + core_rise * (s - 0.5) for s in shifts])
    knots.extend([x_tb + 0.5 * w_tb + sol_rise * (s + 0.5) for s in shifts])
    knots = array(knots)
    knots.sort()
    return knots
