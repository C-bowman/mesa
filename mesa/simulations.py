from pathlib import Path
from abc import ABC, abstractmethod
from typing import Literal, Any


RunStatus = Literal["running", "complete", "timed-out", "crashed"]


class SimulationRun(ABC):
    parameters: dict[str, float]
    directory: Path
    run_number: int
    launch_time: float

    @abstractmethod
    def status(self) -> RunStatus:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def cancel(self):
        pass

    @abstractmethod
    def get_results(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Simulation(ABC):

    @abstractmethod
    def launch(
        self,
        run_number: int,
        simulations_directory: Path,
        parameters: dict,
    ) -> SimulationRun:
        pass

    # def create_case_directory(
    #     self,
    #     run_number: int,
    #     simulations_directory: Path,
    #     input_files: list[str],
    #     filename_map: dict[str, str] = None,
    # ):
    #     case_dir = simulations_directory / f"run_{run_number}"
    #     self.simulations_directory = simulations_directory
    #     filename_map = {} if filename_map is None else filename_map
    #     # create the case directory and copy all the reference files
    #     case_dir.mkdir()
    #     for file_name in input_files:
    #         f = self.simulations_directory / file_name
    #         if f.is_file():
    #             case_name = filename_map.get(file_name, file_name)
    #             subprocess.run(["cp", f, case_dir / case_name])
