import os
from os.path import isfile
from time import time
import logging
import subprocess
from dataclasses import dataclass
from numpy import sum

from sims.interface import SolpsInterface
from sims.likelihoods import gaussian_likelihood, cauchy_likelihood, laplace_likelihood, logistic_likelihood

from mesa.simulations.simulation import Simulation

@dataclass(frozen=True)
class SolpsRun:
    run_id: str
    directory: str
    parameters: dict
    iteration: int
    launch_time: float
    timeout_hours: float

    def status(self):
        whoami = subprocess.run("whoami", capture_output=True, encoding="utf-8")
        username = whoami.stdout.rstrip()

        squeue = subprocess.run(["squeue", "-u", username], capture_output=True, encoding="utf-8")
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

    def cancel(self):
        subprocess.run(["scancel", self.run_id])

    def __key(self):
        return self.run_id, self.directory, self.iteration, self.launch_time

    def __hash__(self):
        return hash(self.__key())

class SOLPS(Simulation):
    def __init__(self,
        exe=None,
        n_proc=1,
        timeout_hours=24, 
        concurrent_runs=1, 
        set_divertor_transport=True,
        transport_profile_bounds=(-0.250, 0.240)
    ):
        super().__init__(exe,n_proc,timeout_hours,concurrent_runs)
        self.set_divertor_transport = set_divertor_transport
        self.transport_profile_bounds = transport_profile_bounds

    def write_solps_transport_inputfile(
        filename,
        grid_dperp,
        values_dperp,
        grid_chieperp,
        values_chieperp,
        grid_chiiperp,
        values_chiiperp,
        set_ana_visc_dperp=True,
        no_pflux=True,
        no_div=False
    ):
        """
        Writes a b2.transport.inputfile to prepare a SOLPS-ITER run.
        Inputs:
            - filename: name of file to output
            - n_species: number of species to apply transport coefficients to
            - grid_dperp: grid where profile of anomalous radial particle diffusion is specified.
            - values_dperp: profile of anomalous radial particle diffusion
            - grid_chieperp: grid where profile of anomalous radial electron heat diffusion is specified.
            - values_chieperp: profile of anomalous radial electron heat diffusion.
            - grid_chiiperp: grid where profile of anomalous radial ion heat diffusion is specified.
            - values_chiiperp: profile of anomalous raidal ion heat diffusion
            - set_ana_visc_dperp: if true, sets profile of anomalous viscosity to be equal to the anomalous particle diffusion.
            - no_pflux: if true, anomalous transport profiles do not apply in the private flux region(s)
            - no_div: if true, anomalous transport profiles do not apply in the divertor(s)
        """

        # Define list to write the file lines to
        sout = [' &TRANSPORT']

        def build_profile(grid, values, code):
            strings = [f' ndata(1, {code} , 1 )= {len(grid)} ,']
            for i, (x, y) in enumerate(zip(grid, values)):
                line = f' tdata(1,{i + 1:2.0f} , {code} , 1 )= {x:6.6f} , tdata(2,{i + 1:2.0f} , {code} , 1 )= {y:6.6f} ,'
                strings.append(line)
            return strings

        sout.extend(  # Write the anomalous radial particle diffusivity profile
            build_profile(grid_dperp, values_dperp, code=1)
        )

        if set_ana_visc_dperp:
            sout.extend(  # If requested, write the anomalous viscosity
                build_profile(grid_dperp, values_dperp, code=7)
            )

        sout.extend(  # Write the anomalous radial electron heat diffusivity profile
            build_profile(grid_chieperp, values_chieperp, code=3)
        )

        sout.extend(  # Write the anomalous radial ion heat diffusivity profile
            build_profile(grid_chiiperp, values_chiiperp, code=4)
        )

        # TODO - check with james whether '9' is intentionally missing here
        extra_species = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        sout.extend([f' addspec({i},{j},1)={i} ,' for i in extra_species for j in [1, 3]])

        if no_pflux:
            sout.append(' no_pflux=.true. ')

        if no_div:
            sout.append(' no_div=.true. ')

        sout.append(' /')

        # Write the list to a file
        with open(filename, 'w') as f:
            for item in sout:
                f.write("%s\n" % item)


    def build_solps_case(
            reference_directory,
            case_directory,
            parameter_dictionary
    ):

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

    def launch(
        iteration,
        directory,
        parameters
    ) -> SolpsRun:
        """
        Evaluates SOLPS for the provided transport profiles and saves the results.

        :param int iteration: \
            The iteration number corresponding to the requested solps-run, used to name
            directory in which the solps output is stored.

        :param str reference_directory: \

        """
        input_files = [
            "input.dat",
            "b2.neutrals.parameters",
            "b2.boundary.parameters",
            "b2.numerics.parameters",
            "b2.transport.parameters",
            "b2mn.dat"
        ]

        self.create_case_directory(iteration,directory,input_files)

        self.build_solps_case(
            reference_directory=directory,
            case_directory=self.case_dir,
            parameter_dictionary=parameters
        )

        # produce transport profiles defined by new point
        chi_params = [parameters[k] for k in self.conductivity_profile]
        chi_radius = self.profile_radius_axis(chi_params, self.transport_profile_bounds)
        chi = self.linear_transport_profile(chi_radius, chi_params, self.transport_profile_bounds)

        D_params = [parameters[k] for k in self.diffusivity_profile]
        D_radius = self.profile_radius_axis(D_params, self.transport_profile_bounds)
        D = self.linear_transport_profile(D_radius, D_params, self.transport_profile_bounds)

        self.write_solps_transport_inputfile(
            filename=self.case_dir + 'b2.transport.inputfile',
            grid_dperp=D_radius,
            values_dperp=D,
            grid_chieperp=chi_radius,
            values_chieperp=chi,
            grid_chiiperp=chi_radius,
            values_chiiperp=chi,
            set_ana_visc_dperp=False,  # TODO - may need to be chosen via settings file
            no_pflux=True,
            no_div=self.set_divertor_transport
        )

        # Go to the SOLPS run directory to prepare execution
        os.chdir(self.case_dir)

        findstr = 'Submitted batch job'

        if n_proc == 1:
            start_run = subprocess.Popen("itmsubmit", stdout=subprocess.PIPE, shell=True)
        if n_proc > 1:
            start_run = subprocess.Popen(f'itmsubmit -m "-np {n_proc}"', stdout=subprocess.PIPE, shell=True)

        start_run_output = start_run.communicate()[0]
        os.chdir(reference_directory)
        # Find the batch job number
        tmp = str(start_run_output).find(findstr)
        job_id = str(start_run_output)[tmp+len(findstr)+1:-3]

        logging.info(f'[solps_interface] Submitted job {job_id}')
        return SolpsRun(
            run_id=job_id,
            directory=self.case_dir,
            iteration=iteration,
            parameters=parameters,
            launch_time=time(),
            timeout_hours=timeout_hours
        )


    def evaluate_log_posterior(diagnostics, directory=None, filename=None) -> dict:
        """
        :param list diagnostics: \
            A list of instrument objects from ``sims.instruments``.

        :param str directory: \
            Path to the directory in which the solps results are stored.

        :param str filename: \
            Filename of the SOLPS balance file, "balance.nc" is used if not specified.

        :return: The posterior log-probability
        """

        # build the path of the solps output file
        fname = "balance.nc" if filename is None else filename
        solps_path = directory + filename  # TODO - make sure this file name is right

        # read the SOLPS data
        solps_data = SolpsInterface(solps_path)

        # update the diagnostics with the latest SOLPS data
        for dia in diagnostics:
            dia.update_interface(solps_data)

        # calculate the log-probabilities
        gaussian_logprob = sum([dia.log_likelihood(likelihood=gaussian_likelihood) for dia in diagnostics])
        cauchy_logprob = sum([dia.log_likelihood(likelihood=cauchy_likelihood) for dia in diagnostics])
        laplace_logprob = sum([dia.log_likelihood(likelihood=laplace_likelihood) for dia in diagnostics])
        logistic_logprob = sum([dia.log_likelihood(likelihood=logistic_likelihood) for dia in diagnostics])
        return {
            "gaussian_logprob": gaussian_logprob,
            "cauchy_logprob": cauchy_logprob,
            "laplace_logprob": laplace_logprob,
            "logistic_logprob": logistic_logprob,
        }

    conductivity_profile = (
        "chi_boundary_left",
        "chi_boundary_right",
        "chi_frac_left",
        "chi_frac_right",
        "chi_barrier_centre",
        "chi_barrier_height",
        "chi_barrier_width",
        "chi_gap_left",
        "chi_gap_right",
    )

    diffusivity_profile = (
        "D_boundary_left",
        "D_boundary_right",
        "D_frac_left",
        "D_frac_right",
        "D_barrier_centre",
        "D_barrier_height",
        "D_barrier_width",
        "D_gap_left",
        "D_gap_right",
    )

    required_parameters = {
        "chi_boundary_left",
        "chi_boundary_right",
        "chi_frac_left",
        "chi_frac_right",
        "chi_barrier_centre",
        "chi_barrier_height",
        "chi_barrier_width",
        "chi_gap_left",
        "chi_gap_right",
        "D_boundary_left",
        "D_boundary_right",
        "D_frac_left",
        "D_frac_right",
        "D_barrier_centre",
        "D_barrier_height",
        "D_barrier_width",
        "D_gap_left",
        "D_gap_right",
    }

    divertor_transport = ("D_div", "chi_div")

    dataframe_columns = (
        "iteration",
        "gaussian_logprob",
        "cauchy_logprob",
        "laplace_logprob",
        "logistic_logprob",
        "error_model",
        "cross_validation",
        "acquisition_function",
        "prediction_mean",
        "prediction_error",
        "convergence_metric",
    )

    from numpy import array, exp, piecewise, concatenate, sort

    def left_section_exp(x, h0, h1, lam, x0):
        return (1 - exp(lam*(x-x0)))*(h0 - h1) + h1


    def middle_section(x, h):
        return h


    def right_section_exp(x, h0, h1, lam, x0):
        return (1 - exp(-lam*(x-x0)))*(h0 - h1) + h1


    def linear_section(x, m, c):
        return m*x + c

    def exponential_transport_profile(x, params):
        L_asymp = params[0]  # left asymptote height
        R_asymp = params[1]  # right asymptote height
        L_shape = params[2]  # left-side decay rate
        R_shape = params[3]  # right-side decay rate
        L_bound = params[4] - params[5]  # boundary point between left and middle sections
        R_bound = params[4] + params[5]  # boundary point between middle and right sections
        M_height= params[6]  # height of the middle section

        in_left = x < L_bound
        in_right = x > R_bound
        in_middle = ~(in_left | in_right)
        conditions = [in_left, in_middle, in_right]
        functions = [
            lambda x: left_section_exp(x, L_asymp, M_height, L_shape, L_bound),
            lambda x: middle_section(x, M_height),
            lambda x: right_section_exp(x, R_asymp, M_height, R_shape, R_bound)
        ]

        return piecewise(x, conditions, functions)


    def linear_profile_knots(params, boundaries):
        knot_locations = array([
            boundaries[0],  # left edge
            boundaries[1],  # right edge
            params[4] - 0.5*params[6],
            params[4] + 0.5*params[6],
            params[4] - 0.5*params[6] - params[7],
            params[4] + 0.5*params[6] + params[8],
        ])

        knot_values = array([
            params[0] + params[5],
            params[1] + params[5],
            params[5],
            params[5],
            params[5] + params[2]*params[0],
            params[5] + params[3]*params[1]
        ])
        sorter = knot_locations.argsort()
        return knot_locations[sorter], knot_values[sorter]


    def linear_transport_profile(x, params, boundaries=(-0.1, 0.1)):
        M_height = params[5]  # height of the middle section
        L_asymp = params[0] + M_height  # left boundary height
        R_asymp = params[1] + M_height  # right asymptote height
        L_mid = params[2]*params[0] + M_height  # left-middle height
        R_mid = params[3]*params[1] + M_height  # left-middle height
        L_bound = params[4] - 0.5*params[6]  # boundary point between left and middle sections
        R_bound = params[4] + 0.5*params[6]  # boundary point between middle and right sections
        L_gap = params[7]  # left-mid-point gap from LM boundary
        R_gap = params[8]  # right-mid-point gap from MR boundary

        # construct boolean arrays of the conditions
        in_left = x < L_bound - L_gap
        in_right = x > R_bound + R_gap
        in_left_mid = (x < L_bound) & (x >= L_bound - L_gap)
        in_right_mid = (x > R_bound) & (x <= R_bound + R_gap)
        in_middle = ~(in_left | in_right | in_left_mid | in_right_mid)
        conditions = [in_left, in_left_mid, in_middle, in_right_mid, in_right]

        # get the line gradients for each section
        Lx, Rx = boundaries
        L_m = (L_mid - L_asymp) / (L_bound - L_gap - Lx)
        R_m = (R_asymp - R_mid) / (Rx - R_bound - R_gap)
        ML_m = (M_height - L_mid) / L_gap
        MR_m = (R_mid - M_height) / R_gap

        # get the line y-intercepts for each section
        L_c = L_asymp - L_m*Lx
        R_c = R_asymp - R_m*Rx
        ML_c = L_mid - ML_m*(L_bound - L_gap)
        MR_c = R_mid - MR_m*(R_bound + R_gap)

        # build functions for each section
        functions = [
            lambda x: linear_section(x, L_m, L_c),
            lambda x: linear_section(x, ML_m, ML_c),
            lambda x: middle_section(x, M_height),
            lambda x: linear_section(x, MR_m, MR_c),
            lambda x: linear_section(x, R_m, R_c)
        ]

        return piecewise(x, conditions, functions)


    def profile_radius_axis(params, boundaries):
        r, _ = linear_profile_knots(params, boundaries)
        # find the spacing that we'll use to place points around each knot
        dr = [r[1] - r[0]]
        for i in range(1, 5):
            dr.append(min(r[i] - r[i-1], r[i+1] - r[i]))
        dr.append(r[5] - r[4])
        # generate the points around each knot
        spacing = array([-6., -2., -1., 1., 2., 6.]) * 0.03
        left_edge = spacing[3:]*dr[0] + r[0]
        right_edge = spacing[:3]*dr[-1] + r[-1]
        middles = [spacing*dr[i] + r[i] for i in range(1, 5)]
        # combine all the points and return them sorted
        return sort(concatenate([left_edge, *middles, right_edge, boundaries]))