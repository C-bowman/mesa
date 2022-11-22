from abc import ABC
from numpy import sum
import logging
from sims.likelihoods import gaussian_likelihood, cauchy_likelihood, laplace_likelihood, logistic_likelihood

class Diagnostic(ABC):
    """
    Abstract base class for all diagnostic objects
    """

    @abstractmethod
    def update_interface(self,data):
        pass
    
    @abstractmethod
    def log_likelihood(self,likelihood=None):
        pass

class WeightedObjectiveFunction:
    diagnostic : list(Diagnostic)

    def __init__(self,diagnostics=None,weights=None):
        self.diagnostics = diagnostics
        self.weights = weights
        if (self.diagnostics is None):
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> No diagnostics were provided to WeightedObjectiveFunction,
                >> but at least one is required.
                """
            )
        if (self.weights is None):
            self.weights = [1.0 for dia in self.diagnostics]
        if len(self.weights) is not len(self.diagnostics):
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> In WeightedObjectiveFunction, the number of diagnostics specified 
                >> is {len(self.diagnostics)}, but the number of weights specified in
                >> is {len(self.weights)}. These must be the same.
                """
            )
    
    def evaluate(self, simulation_interface=si):
        # update the diagnostics with the latest data
        for dia in self.diagnostics:
            dia.update_interface(si)

        return sum([wgt*dia.log_likelihood(likelihood=gaussian_likelihood) for dia,wgt in zip(self.diagnostics,self.weights)])
