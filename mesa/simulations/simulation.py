import subprocess
from os.path import isfile
from abc import ABC

class Simulation(ABC):
    def __init__(self,
        exe=None,
        n_proc=1
    ):
        self.exe = exe
        self.n_proc = n_proc
        self.timeout_hours = timeout_hours
        self.concurrent_runs = concurrent_runs
        self.output_filename = None # should be overwritten by derived classes

    @abstractmethod
    def get_data(self,path=path):
        pass

    def create_case_directory(self,iteration,ref_directory,inputfiles):
        self.casedir = ref_directory + f"iteration_{iteration}/"

        # create the case directory and copy all the reference files
        subprocess.run(["mkdir", self.casedir])
        subprocess.run(["cp", ref_directory + "b2fstate", self.casedir + "b2fstati"])
        for input in inputfiles:
            if isfile(ref_directory + input):
                subprocess.run(["cp", ref_directory + input, self.casedir + input])