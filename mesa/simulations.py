from dataclasses import dataclass
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod
from typing import Literal, Any


RunStatus = Literal["running", "complete", "timed-out", "crashed"]


@dataclass(frozen=True)
class SimulationRun(ABC):
    run_id: str
    directory: Path
    parameters: dict[str, Any]
    run_number: int
    launch_time: float
    timeout_hours: float
    job_queue = None

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

    def __key(self):
        return self.directory, self.run_number, self.launch_time

    def __hash__(self):
        return hash(self.__key())


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
