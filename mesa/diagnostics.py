from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    """
    Abstract base class for all diagnostic objects
    """

    @abstractmethod
    def evaluate(self, simulation_interface) -> dict[str, float]:
        pass


class WeightedObjective(ObjectiveFunction):
    def __init__(self, objectives: list[ObjectiveFunction], weights: list[float] = None):
        self.objectives = objectives
        self.weights = weights

        if self.weights is None:
            self.weights = [1.0 for _ in self.objectives]

        if len(self.weights) != len(self.objectives):
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> In WeightedObjectiveFunction, the number of diagnostics specified 
                >> is {len(self.objectives)}, but the number of weights specified in
                >> is {len(self.weights)}. These must be the same.
                """
            )

    def evaluate(self, simulation_interface) -> dict[str, float]:
        return {
            "objective_value": sum(
                weight * objective.evaluate(simulation_interface)["objective_value"]
                for objective, weight in zip(self.objectives, self.weights)
            )
        }


class Spectrum(Diagnostic):
    def __init__(self, frequency: float, goal: str):
        self.frequency = frequency
        self.goal = goal

        if self.goal not in ["minimize", "maximize"]:
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

    def update_interface(self, data):
        dataslice = [0.5, 0.5, 0.5]
        self.E = data.getE(dataslice)
        self.t = data.getTime()
        self.ff = np.fft.fft(E, axis=t)
        self.freqaxis = np.fft.fftfreq(t, d=self.t[1] - self.t[0])

    def log_likelihood(self, likelihood=None) -> float:
        ilow = 0
        ihigh = len(self.freqaxis)
        for i, f in enumerate(self.freqaxis):
            if f < self.frequency and ilow < i:
                ilow = i
            elif f > self.frequency and ihigh > i:
                ihigh = i
        wgt = (self.freqaxis[ihigh] - self.frequency) / (
            self.freqaxis[ihigh] - self.freqaxis[ilow]
        )
        content = wgt * self.ff[ilow] + (1.0 - wgt) * self.ff[ihigh]

        if self.goal == "maximize":
            return 1.0 - content
        else:
            return content
