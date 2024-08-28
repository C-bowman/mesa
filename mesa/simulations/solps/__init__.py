import os
from os.path import isfile
from time import time
import logging
import subprocess
from dataclasses import dataclass
from numpy import sum

from mesa.simulations import RunStatus, Simulation, SimulationRun
from .transport import write_solps_transport_inputfile
from .models import linear_transport_profile, profile_radius_axis
from .parameters import conductivity_profile, diffusivity_profile, required_parameters
from sims.interface import SolpsInterface
from sims.instruments import Instrument
from sims.likelihoods import gaussian_likelihood, cauchy_likelihood, laplace_likelihood, logistic_likelihood


@dataclass(frozen=True)
class SolpsRun(SimulationRun):

    def status(self) -> RunStatus:
        whoami = subprocess.run("whoami", capture_output=True, encoding="utf-8")
        username = whoami.stdout.rstrip()

        squeue = subprocess.run(
            ["squeue", "-u", username], capture_output=True, encoding="utf-8"
        )
        job_queue = squeue.stdout

        if self.run_id not in job_queue:
            balance_created = isfile(self.directory + "balance.nc")
            status = "complete" if balance_created else "crashed"
        elif (time() - self.launch_time) > self.timeout_hours * 3600.:
            status = "timed-out"
        else:
            status = "running"
        return status

    def cleanup(self):
        output_files = [f for f in os.listdir(self.directory) if isfile(self.directory + f)]
        allowed_files = (
            "balance.nc", "input.dat", "b2.neutrals.parameters",
            "b2.boundary.parameters", "b2.numerics.parameters",
            "b2.transport.parameters", "b2.transport.inputfile",
            "b2mn.dat"
        )
        deletions = [f for f in output_files if f not in allowed_files]
        [os.remove(self.directory + f) for f in deletions]

    def get_results(self) -> SolpsInterface:
        results_path = self.directory + "balance.nc"
        return SolpsInterface(balance_filepath=results_path)

    def cancel(self):
        subprocess.run(["scancel", self.run_id])

    def __key(self):
        return self.run_id, self.directory, self.run_number, self.launch_time

    def __hash__(self):
        return hash(self.__key())


class Solps(Simulation):
    def launch(
        self,
        run_number: int,
        reference_directory: str,
        parameters: dict,
        transport_profile_bounds: tuple[float, float],
        set_div_transport=False,
        n_proc=1,
        timeout_hours=24
    ) -> SolpsRun:
        """
        Evaluates SOLPS for the provided transport profiles and saves the results.

        :param int run_number: \
            The run number corresponding to the requested SOLPS run, used to name
            directory in which the SOLPS output is stored.

        :param str reference_directory: \

        """
        case_dir = reference_directory + f"run_{run_number}/"

        build_solps_case(
            reference_directory=reference_directory,
            case_directory=case_dir,
            parameter_dictionary=parameters
        )

        # produce transport profiles defined by new point
        chi_params = [parameters[k] for k in conductivity_profile]
        chi_radius = profile_radius_axis(chi_params, transport_profile_bounds)
        chi = linear_transport_profile(chi_radius, chi_params, transport_profile_bounds)

        D_params = [parameters[k] for k in diffusivity_profile]
        D_radius = profile_radius_axis(D_params, transport_profile_bounds)
        D = linear_transport_profile(D_radius, D_params, transport_profile_bounds)

        write_solps_transport_inputfile(
            filename=case_dir + 'b2.transport.inputfile',
            grid_dperp=D_radius,
            values_dperp=D,
            grid_chieperp=chi_radius,
            values_chieperp=chi,
            grid_chiiperp=chi_radius,
            values_chiiperp=chi,
            set_ana_visc_dperp=False,  # TODO - may need to be chosen via settings file
            no_pflux=True,
            no_div=set_div_transport
        )

        # Go to the SOLPS run directory to prepare execution
        os.chdir(case_dir)

        findstr = 'Submitted batch job'

        if n_proc == 1:
            start_run = subprocess.Popen("itmsubmit", stdout=subprocess.PIPE, shell=True)
        if n_proc > 1:
            start_run = subprocess.Popen(f'itmsubmit -m "-np {n_proc}"', stdout=subprocess.PIPE, shell=True)

        start_run_output = start_run.communicate()[0]
        os.chdir(reference_directory)
        # Find the batch job number
        tmp = str(start_run_output).find(findstr)
        job_id = str(start_run_output)[tmp + len(findstr) + 1:-3]

        logging.info(f'[solps_interface] Submitted job {job_id}')
        return SolpsRun(
            run_id=job_id,
            directory=case_dir,
            run_number=run_number,
            parameters=parameters,
            launch_time=time(),
            timeout_hours=timeout_hours
        )


def build_solps_case(
    reference_directory: str,
    case_directory: str,
    parameter_dictionary: dict
):
    input_files = [
        "input.dat",
        "b2.neutrals.parameters",
        "b2.boundary.parameters",
        "b2.numerics.parameters",
        "b2.transport.parameters",
        "b2mn.dat"
    ]

    # create the case directory and copy all the reference files
    subprocess.run(["mkdir", case_directory])
    subprocess.run(["cp", reference_directory + "b2fstate", case_directory + "b2fstati"])
    for input in input_files:
        if isfile(reference_directory + input):
            subprocess.run(["cp", reference_directory + input, case_directory + input])

    optional_params = {k for k in parameter_dictionary.keys()} - required_parameters
    written_params = set()

    mesa_input_files = [f + ".mesa" for f in input_files]
    mesa_input_files = [f for f in mesa_input_files if isfile(reference_directory + f)]

    if len(optional_params) != 0:
        if len(mesa_input_files) == 0:
            raise FileNotFoundError(
                f"""
                [ MESA ERROR ]
                >> The following optional parameters were specified
                >> {optional_params}
                >> However no input files with a .mesa extension were
                >> found in the reference directory.
                """
            )

        for mif in mesa_input_files:
            output = []
            with open(reference_directory + mif) as f:
                for line in f:
                    for p in optional_params:
                        if '{'+p+'}' in line:
                            val_string = parameter_dictionary[p]  # TODO - CONVERT TO STRING WITH FORMATTING
                            line.replace('{'+p+'}', val_string)
                            written_params.add(p)
                    output.append(line)

            with open(case_directory + mif, 'w') as f:  # TODO - does this overwrite?
                for item in output:
                    f.write("%s\n" % item)

        unwritten_params = optional_params - written_params
        if len(unwritten_params) > 0:
            raise ValueError(
                f"""
                [ MESA ERROR ]
                >> The following optional parameters were specified
                >> {unwritten_params}
                >> but were not found in any input files with a .mesa extension.
                """
            )


class SolpsLikelihood:
    def __init__(self, diagnostics: list[Instrument]):
        self.diagnostics = diagnostics

    def evaluate(self, simulation_interface: SolpsInterface) -> dict[str, float]:
        # update the diagnostics with the latest SOLPS data
        for dia in self.diagnostics:
            dia.update_interface(simulation_interface)

        # calculate the log-probabilities
        gaussian_logprob = sum([dia.log_likelihood(likelihood=gaussian_likelihood) for dia in self.diagnostics])
        cauchy_logprob = sum([dia.log_likelihood(likelihood=cauchy_likelihood) for dia in self.diagnostics])
        laplace_logprob = sum([dia.log_likelihood(likelihood=laplace_likelihood) for dia in self.diagnostics])
        logistic_logprob = sum([dia.log_likelihood(likelihood=logistic_likelihood) for dia in self.diagnostics])
        return {
            "gaussian_logprob": gaussian_logprob,
            "cauchy_logprob": cauchy_logprob,
            "laplace_logprob": laplace_logprob,
            "logistic_logprob": logistic_logprob,
            "objective_value": cauchy_logprob,
        }
