from mesa.models import linear_transport_profile
from numpy import array


def test_linear_transport_profile():
    bounds = (-0.2, 0.2)
    params = [
        2.,  # left boundary height
        1.,  # right boundary height
        0.7,
        0.6,
        0.01,
        0.3,  # barrier height
        0.02,
        0.1,
        0.02
    ]

    test_locations = array([
        bounds[0],  # left edge
        bounds[1],  # right edge
        params[4] - 0.5*params[6],
        params[4] + 0.5*params[6],
        params[4] - 0.5*params[6] - params[7],
        params[4] + 0.5*params[6] + params[8],
    ])

    test_values = array([
        params[0] + params[5],
        params[1] + params[5],
        params[5],
        params[5],
        params[5] + params[2]*params[0],
        params[5] + params[3]*params[1]
    ])
    test_predictions = linear_transport_profile(test_locations, params, boundaries=bounds)
    assert (test_predictions == test_values).all()
