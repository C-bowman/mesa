class Driver:
    """
    Base class for all drivers, holds list of parameters and their
    limits in a single dictionary. Also holds the simulation
    object that it will launch with parameter values
    """
    parameters_limits = {}

    __init__(self, sim, params):
        self.simulation = sim
        self.parameter_limits = params

    def run():
        return