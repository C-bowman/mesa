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


@dataclass
class HimmelblauRun(SimulationRun):
    process: Process
    parameters: dict[str, float]
    run_number: int
    directory: Path
    launch_time: int

    def status(self) -> RunStatus:
        if self.process.is_alive():
            return "running"

        results_file = self.directory / "himmelblau_output.npy"
        if results_file.is_file():
            return "complete"
        else:
            return "crashed"

    def get_results(self) -> dict[str, float]:
        results_path = self.directory / "himmelblau_output.npy"
        result = load(results_path)
        return {"density": float(result)}

    def cleanup(self):
        pass

    def cancel(self):
        self.process.terminate()

    def __hash__(self):
        return hash((self.run_number, self.launch_time, self.directory))


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
        output_filepath = case_directory / "himmelblau_output.npy"
        savez(input_filepath, **parameters)

        process = Process(
            target=run_himmelblau,
            args=(input_filepath, output_filepath, self.sleep_time),
        )
        process.start()

        return HimmelblauRun(
            process=process,
            parameters=parameters,
            run_number=run_number,
            directory=case_directory,
            launch_time=time(),
        )


class MaximiseDensity(ObjectiveFunction):
    def __init__(self):
        self.name = "density"

    def evaluate(self, simulation_results: dict[str, Any]) -> dict[str, float]:
        return {self.name: simulation_results["density"]}



if __name__ == "__main__":
    from pandas import read_hdf
    from numpy import maximum
    import matplotlib.pyplot as plt

    from inference.gp import SquaredExponential, ConstantMean, ExpectedImprovement
    from mesa.strategies import GPOptimizer
    from mesa.core import Mesa


    himmelblau_sim = Himmelblau(sleep_time=2)
    density_objective = MaximiseDensity()
    gp_strategy = GPOptimizer(
        covariance_kernel=SquaredExponential(),
        mean_function=ConstantMean(),
        acquisition_function=ExpectedImprovement(),
        initial_sample_count=10,
        cross_validation=True,
        trust_region_width=None,
    )

    parameters = {
        "x": (-6.0, 6.0),
        "y": (-6.0, 6.0)
    }

    sim_directory = Path("/home/chris/mesa_testing/himmelblau")
    eval_file = Path("/home/chris/mesa_testing/himmelblau/himmel_test.h5")

    mesa = Mesa(
        parameters=parameters,
        simulation=himmelblau_sim,
        objective_function=density_objective,
        strategy=gp_strategy,
        simulations_directory=sim_directory,
        evaluations_filepath=eval_file,
        max_concurrent_runs=4,
        max_iterations=60
    )

    mesa.run()



    # evaluate the density on a grid
    x = linspace(-6, 6, 128)
    y = linspace(-6, 6, 128)
    density_grid = himmelblau_density(x[:, None], y[None, :], s=12.0)

    # load the simulation data
    df = read_hdf(eval_file, key="evaluations")
    running_max = maximum.accumulate(df["density"])
    print(df)

    # plot the results
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 3]}, figsize=(10, 8)
    )
    plt.subplots_adjust(hspace=0.05)

    ax1.plot(
        df["run_number"],
        df["density"],
        marker="o",
        ls="none",
        c="C0",
        label="objective value",
        markerfacecolor="none"
    )
    ax1.plot(df["run_number"], running_max, c="red", label="best observed")
    ax1.plot(
        [0, 80],
        [1.0, 1.0],
        ls="dashed",
        label="actual max",
        c="black",
    )
    ax1.set_xlabel("function evaluations")
    ax1.set_xlim([0, 70])
    ax1.set_ylim([0, 1.1])
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.set_label_position("right")
    ax1.xaxis.tick_top()
    ax1.set_yticks([])
    ax1.legend(loc=4)

    ax2.contourf(x, y, density_grid.T, 64)
    ax2.plot(df["x"], df["y"], marker="x", color="red", ls="none", markersize=7)
    plt.tight_layout()
    plt.show()