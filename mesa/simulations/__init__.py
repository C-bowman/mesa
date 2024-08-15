from dataclasses import dataclass
import subprocess
from os.path import isfile
from abc import ABC, abstractmethod
from typing import Literal


RunStatus = Literal["running", "complete", "timed-out", "crashed"]


@dataclass(frozen=True)
class SimulationRun(ABC):
    run_id: str
    directory: str
    parameters: dict
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

    def cancel(self):
        subprocess.run(["scancel", self.run_id])

    def __key(self):
        return self.run_id, self.directory, self.iteration, self.launch_time

    def __hash__(self):
        return hash(self.__key())


class Simulation(ABC):
    exe: str
    n_proc: int
    timeout_hours: float
    output_filename: str
    reference_dir: str

    def __init__(self, exe=None, n_proc=1, timeout_hours=1):
        self.exe = exe
        self.n_proc = n_proc
        self.timeout_hours = timeout_hours

    @abstractmethod
    def launch(
        self, run_number: int, directory: str, parameters: dict
    ) -> SimulationRun:
        pass

    @abstractmethod
    def get_data(self, path: str):
        pass

    def create_case_directory(
        self,
        run_number: int,
        ref_directory: str,
        input_files: list[str],
        filename_map: dict[str, str] = None,
    ):
        self.case_dir = f"{ref_directory}/run_{run_number}/"
        self.reference_dir = ref_directory
        filename_map = {} if filename_map is None else filename_map
        # create the case directory and copy all the reference files
        subprocess.run(["mkdir", self.case_dir])
        for file_name in input_files:
            if isfile(ref_directory + file_name):
                case_name = filename_map.get(file_name, file_name)
                subprocess.run(
                    ["cp", ref_directory + file_name, self.case_dir + case_name]
                )
