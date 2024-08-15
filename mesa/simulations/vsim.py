import logging
import subprocess
import os
from os.path import isfile
from time import time
from dataclasses import dataclass
from mesa.simulations import Simulation, SimulationRun, RunStatus


class VSimInterface:

    def __init__(self, path: str):
        self.path = path

    def get_E(self, dataslice):
        return 0.0

    def get_time(self):
        return 0.0


@dataclass(frozen=True)
class VSimRun(SimulationRun):
    slurm: bool

    def status(self) -> RunStatus:
        if self.slurm:
            super().status()
            if self.slurm and (self.run_id not in self.job_queue):
                output_created = isfile(self.directory + "balance.nc")
                status = "complete" if output_created else "crashed"
            elif (time() - self.launch_time) > self.timeout_hours * 3600.0:
                status = "timed-out"
            else:
                status = "running"
        else:
            status = "complete"

        return status

    def cleanup(self):
        output_files = [
            f for f in os.listdir(self.directory) if isfile(self.directory + f)
        ]
        return

    def __key(self):
        return self.run_id, self.directory, self.run_number, self.launch_time

    def __hash__(self):
        return hash(self.__key())


class VSim(Simulation):
    base_input_file: str
    slurm: bool

    def __init__(
        self, exe=None, n_proc=1, timeout_hours=24, input_file="vsim.pre", slurm=False
    ):
        super().__init__(exe, n_proc, timeout_hours)
        self.slurm = slurm
        self.base_input_file = input_file

    def write_vsim_inputfiles(self, filenames):
        """
        Writes a VSim .pre file changing the desired parameters
        """

        # Write the list to a file
        for f in filenames:
            with open(f, "w+") as f:
                return
                # change desired param values, throw error of one cannot be found

    def launch(self, run_number: int, directory: str, parameters: dict) -> VSimRun:
        """
        Evaluates VSim for the provided parameter values

        :param int run_number: \
            The run number corresponding of the requested simulation run, used to name
            directory in which the simulation output is stored.

        :param str directory: \
            The directory inside which a new directory for the run will be created.

        :param parameters: \
            The list of parameters to set in the input file.

        """
        filename = self.base_input_file.split(".")[0]
        input_files = [filename + ".pre", filename + ".in"]

        self.create_case_directory(
            run_number=run_number, ref_directory=directory, input_files=input_files
        )

        input_paths = [self.case_dir + f for f in input_files]
        self.write_vsim_inputfiles(input_paths)

        # Go to the run directory to prepare execution
        os.chdir(self.case_dir)

        if self.n_proc == 1:
            command = "source " + self.exe + " ; vorpalser -i " + input_files[0]
            logging.info("Executing command: " + command)
            start_run = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        if self.n_proc > 1:
            command = (
                "source "
                + self.exe
                + f" ; mpirun -np {self.n_proc} vorpal -i "
                + input_files[0]
            )
            logging.info("Executing command: " + command)
            start_run = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

        start_run_output = start_run.communicate()[0]
        os.chdir(self.reference_dir)
        # Find the batch job number
        if self.slurm:
            findstr = "r"
            tmp = str(start_run_output).find(findstr)
            job_id = str(start_run_output)[tmp + len(findstr) + 1 : -3]
            logging.info(f"[vsim_interface] Submitted job {job_id}")
        else:
            job_id = 0
            logging.info(f"[vsim_interface] Running job {run_number}")

        return VSimRun(
            run_id=job_id,
            directory=self.case_dir,
            run_number=run_number,
            parameters=parameters,
            launch_time=time(),
            timeout_hours=self.timeout_hours,
            slurm=self.slurm,
        )

    def get_data(self, path: str):
        """
        Returns interface to run data at provided path
        """
        return VSimInterface(path)
