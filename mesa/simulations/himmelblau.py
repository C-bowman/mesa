from numpy import linspace, exp, save, savez, load
import subprocess
from time import sleep
from os.path import isfile
import matplotlib.pyplot as plt
from mesa.simulations import RunStatus, Simulation, SimulationRun
from dataclasses import dataclass


def himmelblau_func(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_density(x, y, a=2e-2, s=15.0):
    z_sqr = ((x + 3.7793)**2 + (y + 3.2832)**2 ) / s**2
    return exp(-a * himmelblau_func(x, y) - 0.5*z_sqr)

x = linspace(-6, 6, 128)
y = linspace(-6, 6, 128)

f = himmelblau_density(x[:, None], y[None, :])

plt.contourf(x, y, f.T, 64)
plt.show()

def run_himmelblau(input_filepath:str, sleep_time: int):
    sleep(sleep_time)
    data = load(input_filepath)
    result = himmelblau_density(
        x=data["x"],
        y=data["y"]
    )
    save("himmmelblau_results.npy", result)


@dataclass(frozen=True)
class HimmelblauRun(SimulationRun):

    def status(self) -> RunStatus:
        results_created = isfile(self.directory + "himmmelblau_results.npy")
        status = "complete" if results_created else "running"
        return status

    def get_results(self) -> dict[str, float]:
        results_path = self.directory + "himmmelblau_results.npy"
        results = load(results_path)
        return {"density": results["density"]}


class Himmelblau(Simulation):
    def launch(
        self,
        run_number: int,
        reference_directory: str,
        parameters: dict,
        *args,
        **kwargs,
    ) -> SimulationRun:
        pass

        case_directory = reference_directory + f"run_{run_number}/"
        # create the case directory and copy all the reference files
        subprocess.run(["mkdir", case_directory])
        savez(case_directory, **parameters)

