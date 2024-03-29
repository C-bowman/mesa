from numpy import array, exp, piecewise, concatenate, sort


def left_section_exp(x, h0, h1, lam, x0):
    return (1 - exp(lam*(x-x0)))*(h0 - h1) + h1


def middle_section(x, h):
    return h


def right_section_exp(x, h0, h1, lam, x0):
    return (1 - exp(-lam*(x-x0)))*(h0 - h1) + h1


def linear_section(x, m, c):
    return m*x + c


def exponential_transport_profile(x, params):
    L_asymp = params[0]  # left asymptote height
    R_asymp = params[1]  # right asymptote height
    L_shape = params[2]  # left-side decay rate
    R_shape = params[3]  # right-side decay rate
    L_bound = params[4] - params[5]  # boundary point between left and middle sections
    R_bound = params[4] + params[5]  # boundary point between middle and right sections
    M_height= params[6]  # height of the middle section

    in_left = x < L_bound
    in_right = x > R_bound
    in_middle = ~(in_left | in_right)
    conditions = [in_left, in_middle, in_right]
    functions = [
        lambda x: left_section_exp(x, L_asymp, M_height, L_shape, L_bound),
        lambda x: middle_section(x, M_height),
        lambda x: right_section_exp(x, R_asymp, M_height, R_shape, R_bound)
    ]

    return piecewise(x, conditions, functions)


def linear_profile_knots(params, boundaries):
    knot_locations = array([
        boundaries[0],  # left edge
        boundaries[1],  # right edge
        params[4] - 0.5*params[6],
        params[4] + 0.5*params[6],
        params[4] - 0.5*params[6] - params[7],
        params[4] + 0.5*params[6] + params[8],
    ])

    knot_values = array([
        params[0] + params[5],
        params[1] + params[5],
        params[5],
        params[5],
        params[5] + params[2]*params[0],
        params[5] + params[3]*params[1]
    ])
    sorter = knot_locations.argsort()
    return knot_locations[sorter], knot_values[sorter]


def linear_transport_profile(x, params, boundaries=(-0.1, 0.1)):
    M_height = params[5]  # height of the middle section
    L_asymp = params[0] + M_height  # left boundary height
    R_asymp = params[1] + M_height  # right asymptote height
    L_mid = params[2]*params[0] + M_height  # left-middle height
    R_mid = params[3]*params[1] + M_height  # left-middle height
    L_bound = params[4] - 0.5*params[6]  # boundary point between left and middle sections
    R_bound = params[4] + 0.5*params[6]  # boundary point between middle and right sections
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
    L_c = L_asymp - L_m*Lx
    R_c = R_asymp - R_m*Rx
    ML_c = L_mid - ML_m*(L_bound - L_gap)
    MR_c = R_mid - MR_m*(R_bound + R_gap)

    # build functions for each section
    functions = [
        lambda x: linear_section(x, L_m, L_c),
        lambda x: linear_section(x, ML_m, ML_c),
        lambda x: middle_section(x, M_height),
        lambda x: linear_section(x, MR_m, MR_c),
        lambda x: linear_section(x, R_m, R_c)
    ]

    return piecewise(x, conditions, functions)


def profile_radius_axis(params, boundaries):
    r, _ = linear_profile_knots(params, boundaries)
    # find the spacing that we'll use to place points around each knot
    dr = [r[1] - r[0]]
    for i in range(1, 5):
        dr.append(min(r[i] - r[i-1], r[i+1] - r[i]))
    dr.append(r[5] - r[4])
    # generate the points around each knot
    spacing = array([-6., -2., -1., 1., 2., 6.]) * 0.03
    left_edge = spacing[3:]*dr[0] + r[0]
    right_edge = spacing[:3]*dr[-1] + r[-1]
    middles = [spacing*dr[i] + r[i] for i in range(1, 5)]
    # combine all the points and return them sorted
    return sort(concatenate([left_edge, *middles, right_edge, boundaries]))


