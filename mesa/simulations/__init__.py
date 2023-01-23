from dataclasses import dataclass
import subprocess
from os.path import isfile
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class SimulationRun(ABC):
    run_id: str
    directory: str
    parameters: dict
    iteration: int
    launch_time: float
    timeout_hours: float
    job_queue = None

    def status(self):
        whoami = subprocess.run("whoami", capture_output=True, encoding="utf-8")
        username = whoami.stdout.rstrip()
        squeue = subprocess.run(
            ["squeue", "-u", username], capture_output=True, encoding="utf-8"
        )
        self.job_queue = squeue.stdout

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
        self.output_filename

    @abstractmethod
    def launch(self, iteration=None, directory=None, parameters=None) -> SimulationRun:
        pass

    @abstractmethod
    def get_data(self, path=None):
        pass

    def create_case_directory(self, iteration, ref_directory, inputfiles):
        self.casedir = ref_directory + f"/iteration_{iteration}/"
        self.reference_dir = ref_directory

        # create the case directory and copy all the reference files
        subprocess.run(["mkdir", self.casedir])
        subprocess.run(["cp", ref_directory + "b2fstate", self.casedir + "b2fstati"])
        for input in inputfiles:
            if isfile(ref_directory + input):
                subprocess.run(["cp", ref_directory + input, self.casedir + input])
