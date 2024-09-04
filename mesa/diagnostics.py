from abc import ABC, abstractmethod


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
