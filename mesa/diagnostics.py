from abc import ABC, abstractmethod
from numpy import sum
import logging
import numpy as np
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
    diagnostic : list
    weights : list

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
    
    def evaluate(self, simulation_interface=None):
        # update the diagnostics with the latest data
        for dia in self.diagnostics:
            dia.update_interface(simulation_interface)

        return {"logprob" : sum([wgt*dia.log_likelihood(likelihood=gaussian_likelihood) for dia,wgt in zip(self.diagnostics,self.weights)])}

class Spectrum(Diagnostic):
    frequency : float
    goal : str

    def __init__(self, frequency=None, goal=None):
        self.frequency = frequency
        self.goal = goal

        if ((self.goal != "minimize") and (self.goal != "maximize")):
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> Spectrum goal should either be "minimize" or "maximize"
                """
            )
        
        if type(self.frequency) is not float:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> Spectrum frequency should be a float
                """
            )

    def update_interface(self,data):
        dataslice = [0.5,0.5,0.5]
        self.E = data.getE(dataslice)
        self.t = data.getTime()
        self.ff = np.fft.fft(E,axis=t)
        self.freqaxis = np.fft.fftfreq(t,d=self.t[1]-self.t[0])

    def log_likelihood(self,likelihood=None) -> float:
        ilow = 0
        ihigh = len(self.freqaxis)
        for i,f in enumerate(self.freqaxis):
            if f < self.frequency and ilow < i :
                ilow = i
            elif f > self.frequency and ihigh > i:
                ihigh = i
        wgt = ( self.freqaxis[ihigh] - self.frequency ) / (self.freqaxis[ihigh] - self.freqaxis[ilow])
        content = wgt * self.ff[ilow] + (1.-wgt) * self.ff[ihigh]
        
        if self.goal == "maximize":
            return 1.-content
        else:
            return content