import subprocess
import os
from os.path import isfile

class Simulation():
    def __init__(self,
        exe=None,
        n_proc=1
    ):
        self.exe = exe
        self.n_proc = n_proc
        self.timeout_hours = timeout_hours
        self.concurrent_runs = concurrent_runs

    def create_case_directory(self,iteration,ref_directory,inputfiles):
        self.casedir = ref_directory + f"iteration_{iteration}/"

        # create the case directory and copy all the reference files
        subprocess.run(["mkdir", self.casedir])
        subprocess.run(["cp", ref_directory + "b2fstate", self.casedir + "b2fstati"])
        for input in inputfiles:
            if isfile(ref_directory + input):
                subprocess.run(["cp", ref_directory + input, self.casedir + input])