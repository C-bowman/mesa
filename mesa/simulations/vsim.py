import os
from os.path import isfile
from time import time
import logging
import subprocess
from dataclasses import dataclass

from mesa.simulations.simulation import Simulation
from mesa.simulations.simulation import SimulationRun

class VSimInterface():
    path : str

    def __init__(self,path):
        self.path = path

@dataclass(frozen=True)
class VSimRun(SimulationRun):

    def status(self):
        super().status()
        if self.run_id not in self.job_queue:
            balance_created = isfile(self.directory + "balance.nc")
            status = "complete" if balance_created else "crashed"
        elif (time() - self.launch_time) > self.timeout_hours * 3600.:
            status = "timed-out"
        else:
            status = "running"
        return status

    def cleanup(self):
        output_files = [f for f in os.listdir(self.directory) if isfile(self.directory + f)]
        return
        # allowed_files = (
        #     "balance.nc", "input.dat", "b2.neutrals.parameters",
        #     "b2.boundary.parameters", "b2.numerics.parameters",
        #     "b2.transport.parameters", "b2.transport.inputfile",
        #     "b2mn.dat"
        # )
        # deletions = [f for f in output_files if f not in allowed_files]
        # [os.remove(self.directory + f) for f in deletions]

class VSim(Simulation):
    base_input_file : str
    slurm : bool

    def __init__(self,
        exe=None,
        n_proc=1,
        timeout_hours=24,
        input_file='vsim.pre',
        slurm=False
    ):
        super().__init__(exe,n_proc,timeout_hours)
        self.slurm = slurm
        self.base_input_file = input_file

    def write_vsim_inputfiles(self,
        filenames
    ):
        """
        Writes a VSim .pre file changing the desired parameters
        """

        # Write the list to a file
        for f in filenames:
            with open(f, 'w+') as f:
                # change desired param values, throw error of one cannot be found

    def launch(self,
        iteration=None,
        directory=None,
        parameters=None
    ) -> VSimRun:
        """
        Evaluates VSim for the provided parameter values

        :param int iteration: \
            The iteration number corresponding to the requested solps-run, used to name
            directory in which the solps output is stored.

        :param str reference_directory: \
        
        :param list parameters:
            The list of parameters to set in the input file \

        """
        filename = self.base_input_file.split('.')[0]
        input_files = [
            filename+".pre",
            filename+".in"
        ]

        self.create_case_directory(iteration,directory,input_files)

        for infile in input_files:
            infile = self.casedir + infile
        self.write_vsim_inputfiles(input_files)

        # Go to the SOLPS run directory to prepare execution
        os.chdir(self.casedir)

        # source
        try:
            src = subprocess.Popen("source "+self.exe, stdout=subprocess.PIPE, shell=True)
        except:
            raise Exception(
                f"""
                [ MESA ERROR ]
                >> Unable to source vorpalall.sh or VSimComposer.sh, please provide
                >>   correct path.
                """
            )
        if n_proc == 1:
            start_run = subprocess.Popen("vorpalser "+input_files[0], stdout=subprocess.PIPE, shell=True)
        if n_proc > 1:
            start_run = subprocess.Popen("mpirun -np vorpal "+input_files[0], stdout=subprocess.PIPE, shell=True)

        start_run_output = start_run.communicate()[0]
        os.chdir(self.reference_dir)
        # Find the batch job number
        if (self.slurm):
            findstr='r'
            tmp = str(start_run_output).find(findstr)
            job_id = str(start_run_output)[tmp+len(findstr)+1:-3]
            logging.info(f'[vsim_interface] Submitted job {job_id}')
        else:
            job_id = 0
            logging.info(f'[vsim_interface] Running job {iteration}')

        return VSimRun(
            run_id=job_id,
            directory=self.casedir,
            iteration=iteration,
            parameters=parameters,
            launch_time=time(),
            timeout_hours=self.timeout_hours
        )

    def get_data(path=None):
        """
        Returns interface to run data at provided path
        """
        if path==None:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> Path not provided for simulation data in VSim get_data()
                """
            )
        return VSimInterface(path)