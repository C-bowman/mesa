from pathlib import Path
from multiprocessing import Process
from typing import Any

from numpy import linspace, exp, save, savez, load
from time import sleep, time
from dataclasses import dataclass

from mesa.simulations import RunStatus, Simulation, SimulationRun
from mesa.objectives import ObjectiveFunction


def himmelblau_func(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def himmelblau_density(x, y, a=2e-2, s=15.0):
    z_sqr = ((x + 3.7793) ** 2 + (y + 3.2832) ** 2) / s**2
    return exp(-a * himmelblau_func(x, y) - 0.5 * z_sqr)


def run_himmelblau(input_filepath: Path, output_filepath: Path, sleep_time: int):
    sleep(sleep_time)
    data = load(input_filepath)
    result = himmelblau_density(x=data["x"], y=data["y"])
    save(output_filepath, result)


@dataclass(frozen=True)
class HimmelblauRun(SimulationRun):
    process: Process
    run_number: int
    directory: Path
    launch_time: int

    def status(self) -> RunStatus:
        if self.process.is_alive():
            return "running"

        results_file = self.directory / "himmmelblau_output.npy"
        if results_file.exists():
            return "complete"
        else:
            return "crashed"

    def get_results(self) -> dict[str, float]:
        results_path = self.directory / "himmmelblau_output.npy"
        results = load(results_path)
        return {"density": results["density"]}

    def cleanup(self):
        pass

    def cancel(self):
        self.process.terminate()


class Himmelblau(Simulation):
    def __init__(self, sleep_time: int):
        self.sleep_time = sleep_time

    def launch(
        self,
        run_number: int,
        simulations_directory: Path,
        parameters: dict,
    ) -> HimmelblauRun:

        case_directory = simulations_directory / f"run_{run_number}"
        case_directory.mkdir()
        # create the case directory and copy all the reference files
        input_filepath = case_directory / "himmelblau_input.npz"
        output_filepath = case_directory / "himmelblau_output.npz"
        savez(input_filepath, **parameters)

        process = Process(
            target=run_himmelblau,
            args=(input_filepath, output_filepath, self.sleep_time),
        )
        process.start()

        return HimmelblauRun(
            process=process,
            run_number=run_number,
            directory=case_directory,
            launch_time=time(),
        )


class MaximiseDensity(ObjectiveFunction):
    def __init__(self):
        pass

    def evaluate(self, simulation_results: dict[str, Any]) -> dict[str, float]:
        return simulation_results["density"]



if __name__ == "__main__":
    x = linspace(-6, 6, 128)
    y = linspace(-6, 6, 128)

    f = himmelblau_density(x[:, None], y[None, :], s=12.0)
    import matplotlib.pyplot as plt

    plt.contourf(x, y, f.T, 64)
    plt.show()
    from mesa.strategies import GPOptimizer
    from mesa.core import Mesa

    himmelblau_sim = Himmelblau(sleep_time=30)
    density_objective = MaximiseDensity()
    gp_strategy = GPOptimizer()