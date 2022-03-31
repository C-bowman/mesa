
from os import chdir
from os.path import exists
from time import sleep, time
import logging
import subprocess
from numpy import sum

from mesa.models import linear_transport_profile, profile_radius_axis
from mesa.parameters import conductivity_profile, diffusivity_profile
from sims.interface import SolpsInterface
from sims.likelihoods import gaussian_likelihood, cauchy_likelihood, laplace_likelihood, logistic_likelihood


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
            line = f' tdata(1,{i + 1:2.0f} , {code} , 1 )= {x:6.3f} , tdata(2,{i + 1:2.0f} , {code} , 1 )= {y:6.3f} ,'
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
    # create the case directory and copy all the reference files
    subprocess.run(["mkdir", case_directory])
    subprocess.run(["cp", reference_directory + "b2fstate", case_directory + "b2fstate"])
    subprocess.run(["cp", reference_directory + "b2fstate", case_directory + "b2fstai"])

    variables = [k for k in parameter_dictionary.keys()]

    input_files = ['b2mn.dat', 'b2.transport.parameters']
    for ifile in input_files:
        output = []
        with open(reference_directory + ifile) as f:
            for line in f:
                for v in variables:
                    if '{'+v+'}' in line:
                        val_string = parameter_dictionary[v] # TODO - CONVERT TO STRING WITH FORMATTING
                        line.replace('{'+v+'}', val_string)
                output.append(line)

        with open(case_directory + ifile, 'w') as f:
            for item in output:
                f.write("%s\n" % item)






def run_solps(
    iteration,
    reference_directory,
    parameter_dictionary,
    transport_profile_bounds,
    set_div_transport=False,
    n_proc=1
):
    """
    Evaluates SOLPS for the provided transport profiles and saves the results.

    :param int iteration: \
        The iteration number corresponding to the requested solps-run, used to name
        directory in which the solps output is stored.

    :param str reference_directory: \

    """
    case_dir = reference_directory + f"{run_id_string}_{iteration}/" # TODO

    build_solps_case(
        reference_directory=reference_directory,
        case_directory=case_dir,
        parameter_dictionary=parameter_dictionary
    )

    # produce transport profiles defined by new point
    radius = profile_radius_axis(boundaries=transport_profile_bounds)

    chi_params = [parameter_dictionary[k] for k in conductivity_profile]
    chi = linear_transport_profile(radius, chi_params, boundaries=transport_profile_bounds)

    D_params = [parameter_dictionary[k] for k in diffusivity_profile]
    D = linear_transport_profile(radius, D_params, boundaries=transport_profile_bounds)

    write_solps_transport_inputfile(
        filename=case_dir + 'b2.transport.inputfile',
        grid_dperp=radius,
        values_dperp=D,
        grid_chieperp=radius,
        values_chieperp=chi,
        grid_chiiperp=radius,
        values_chiiperp=chi,
        set_ana_visc_dperp=True,
        no_pflux=True,
        no_div=set_div_transport
    )


    # Go to the SOLPS run directory to prepare execution
    chdir(case_dir)

    findstr = 'Submitted batch job'

    if n_proc == 1:
        start_run = subprocess.Popen('itmsubmit',stdout=subprocess.PIPE, shell=True)
    if n_proc > 1:
        start_run = subprocess.Popen('itmsubmit -m "-np '+str(n_proc)+'"', stdout=subprocess.PIPE, shell=True)

    start_run_output = start_run.communicate()[0]
    chdir(reference_directory)
    # Find the batch job number
    tmp = str(start_run_output).find(findstr)
    jobstr = str(start_run_output)[tmp+len(findstr)+1:-3]

    logging.info(f'[solps_interface] Submitted job {jobstr}')
    return jobstr


def check_solps_run(job_ids, timeout_hours=10):
    """

    :param job_ids: \
        list of job id strings
    """
    # set a time-out point
    timeout = time() + 3600*timeout_hours

    # wait for run completion
    uname = subprocess.Popen('whoami',stdout=subprocess.PIPE, shell=True)
    username = uname.communicate()[0]
    username = str(username.rstrip(),'utf-8')

    test = subprocess.Popen('squeue -u '+username,stdout=subprocess.PIPE,shell=True)
    jobqueue = test.communicate()[0]

    job_finished = [job not in str(jobqueue) for job in job_ids]

    if all(job_finished):

    while True:
        if all(job_finished): # check if the job is still running
            logging.info(f'[solps_interface] Job {jobstr} has finished')
            break
        else:
            logging.debug('[solps_interface] Job '+jobstr+' is in progress...')

        if time() > timeout: # check to see if we've timed out
            #raise TimeoutError('No SOLPS output file was detected within the time-out limit')
            logging.warning('[solps_interface] No SOLPS output file was detected within the time-out limit')
            cancel_solps(jobstr) # TODO - make this work on a list of job id's
            break

        sleep(30) # wait for 30 seconds between checks

        test = subprocess.Popen('squeue -u '+username, stdout=subprocess.PIPE, shell=True)
        jobqueue = test.communicate()[0]
        job_finished = str(jobqueue).find(jobstr)

    # Run this when SOLPS has finished
    # build the path of the solps output file
    output_path = run_directory + 'balance.nc'
    new_output_path = output_directory + 'solps_run_{}.nc'.format(int(iteration))


    # TODO - replace this with a check to see if balance file is actually there
    if exists(output_path):
        logging.info('[solps_interface] SOLPS run completed successfully.')
        # copy the SOLPS output to the solps_output_directory
        copyoutputfiles = subprocess.Popen('cp '+output_path+' '+new_output_path,stdout=subprocess.PIPE,shell=True)
        copyoutputfiles.communicate()

        # zip up the SOLPS input files
        input_files = 'b2mn.dat b2ah.dat input.dat b2.transport.parameters b2.boundary.parameters b2.neutrals.parameters b2.transport.inputfile'

        zipinputfiles = subprocess.Popen('zip input_files.zip '+input_files,stdout=subprocess.PIPE,shell=True)
        zipinputfiles.communicate()

        # Then copy the SOLPS input files to the solps_output directory
        output_path = run_directory + 'input_files.zip'
        new_output_path = output_directory + 'solps_input_files_{}.zip'.format(int(iteration))
        copyinputfiles = subprocess.Popen('cp '+output_path+' '+new_output_path,stdout=subprocess.PIPE,shell=True)
        copyinputfiles.communicate()

        chdir(orig_dir)

        return True
    else:
        logging.warning('[solps_interface] SOLPS run failed.')

        chdir(orig_dir)

        return False


def cancel_solps(jobstr):
    """
    Cancels a SOLPS run in case of a timeout or other error.  An output file is not written
    by SOLPS, so solps_interface knows the code ended with an error
    """

    test = subprocess.Popen('scancel '+jobstr,stdout=subprocess.PIPE,shell=True)
    comm = test.communicate()[0]


def evaluate_log_posterior(diagnostics, iteration=None, directory=None):
    """
    :param list diagnostics: \
        A list of instrument objects from ``sims.instruments``.

    :param int iteration: \
        iteration number of the solps run for which the posterior log-probability is calculated.

    :param str directory: \
        Path to the directory in which the solps results, diagnostic data and training data
        are stored.

    :param str diagnostic_data_file: \
        File name of the diagnostic data file.

    :return: The posterior log-probability
    """

    # build the path of the solps output file
    solps_path = directory + 'solps_run_{}.nc'.format(int(iteration))

    # read the SOLPS data
    solps_data = SolpsInterface(solps_path)

    # update the diagnostics with the latest SOLPS data
    for dia in diagnostics:
        dia.update_interface(solps_data)

    # calculate the log-probabilities
    gauss_logprob = sum([dia.log_likelihood(likelihood=gaussian_likelihood) for dia in diagnostics])
    cauchy_logprob = sum([dia.log_likelihood(likelihood=cauchy_likelihood) for dia in diagnostics])
    laplace_logprob = sum([dia.log_likelihood(likelihood=laplace_likelihood) for dia in diagnostics])
    logistic_logprob = sum([dia.log_likelihood(likelihood=logistic_likelihood) for dia in diagnostics])
    return gauss_logprob, cauchy_logprob, laplace_logprob, logistic_logprob
