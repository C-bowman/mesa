from collections.abc import Sequence
from abc import ABC, abstractmethod
from numpy import array, ndarray
from pandas import DataFrame


class Strategy(ABC):
    strategy_columns: list[str]

    @abstractmethod
    def propose_evaluations(
        self,
        evaluation_data: DataFrame,
        optimisation_bounds: dict[str, tuple[float, float]],
        objective_name: str,
    ) -> list[dict]:
        pass

    @staticmethod
    def normalise_parameters(
        v: ndarray, bounds: Sequence[tuple[float, float]]
    ) -> ndarray:
        return array([(k - b[0]) / (b[1] - b[0]) for k, b in zip(v, bounds)])

    @staticmethod
    def reverse_normalisation(
        v: ndarray, bounds: Sequence[tuple[float, float]]
    ) -> ndarray:
        return array([b[0] + (b[1] - b[0]) * k for k, b in zip(v, bounds)])
