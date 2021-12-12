
from os import getcwd, chdir
from os.path import exists
from time import sleep, time
import logging
import subprocess
import filecmp
from numpy import sum

from sims.interface import SolpsInterface
from sims.likelihoods import gaussian_likelihood, cauchy_likelihood, laplace_likelihood, logistic_likelihood


def write_solps_transport_inputfile(
    filename,
    n_species,
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






def write_solps_transport_paramters(
    filename,
    n_species,
    dna,
    dpa,
    vla,
    vsa,
    hci,
    hce,
    sig,
    alf
):

    """
    Writes a b2.transport.inputfile to prepare a SOLPS-ITER run.
    Inputs:
        - filename: name of file to output
        - n_species: number of species to apply transport coefficients to
        - dna: anomalous radial particle diffusivity
        - dpa: anomalous pressure driven radial particle diffusivity (usually zero)
        - vla: radial pinch velocity (usually zero)
        - vsa: anomalous viscosity (usually 1.0)
        - hci: anomalous radial ion heat diffusivity
        - hce: anomalous radial electron heat diffusivity
        - sig: anomalous current conductivity
        - alf: anomalous thermo-electric current
    """

    # Prepare the file contents in a list
    # TODO - check with james if the param values need any formatting
    sout = [
        ' &transport',
        f' flag_dna=1, parm_dna={int(n_species)}*{dna},',
        f' flag_dpa=1, parm_dpa={int(n_species)}*{dpa},',
        f' flag_vla=1, parm_vla={int(n_species)}*{vla},',
        f' flag_vsa=1, parm_vsa={int(n_species)}*{vsa},',
        f' flag_hci=1, parm_hci={int(n_species)}*{hci},',
        f' flag_hce=1, parm_hce={hce},',
        f' flag_sig=1, parm_sig={sig},',
        f' flag_alf=1, parm_alf={alf},',
        ' /'
    ]

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)






def write_solps_b2mn_dat(
    filename,
    n_timestep,
    d_timestep,
    cflme=0.3,  # Multiplier to the electron heat flux limit
    cflmi=1.0,  # Multiplier to the ion heat flux limit
    cflmv=0.5,  # Multiplier to the viscous heat flux limit
    cflal=0.5,  # Multilier to the thermo-electric coefficient flux limit
    cflab=0.5,  #  Multiplier to the friction force flux limit
    label='Gaussian process test case'
):

    sout = [
        '*label          (lblmn: character*60)',
        f"  '{label}'",
        '*b2cmpa  basic parameters',
        '*b2cmpb  boundary conditions',
        '*b2cmpt  transport coefficients',
        '*cflim    cflme    cflmi    cflmv     cflal     cflab',
        f"  '-1'        {cflme}      {cflmi}      {cflmv}      {cflal}      {cflab}        0        0        0",
        '*endphy',
        f"'b2mndr_ntim'     '{n_timestep}'",
        f"'b2mndr_dtim'     '{d_timestep}'",
        "'b2mndt_nstg0'    '1'",
        "'b2mndt_nstg1'    '1'",
        "'b2mndt_nstg2'    '10'",
        "'b2mndt_rxf'      '0.5'",
        "'b2mndr_cpu'      '80000.0'",
        "'b2mndr_savecpu'  '1000.0'",
        "'b2mndr_stim'     '-5'",
        "'b2mndr_b2time'   '10'",
        "'b2stbc_boundary_namelist' '1'",
        "'b2stbr_neutrals_namelist' '1'",
        "'b2tqna_transport_namelist' '1'",
        "'b2tqna_inputfile' '1'",
        "'b2tqna_ballooning' '0'",
        "'b2tqna_ballooning_rescale' '1'",
        "'b2mndr_eirene'                   '1'",
        "'b2mndr_rescale_neutrals_sources' '1e-20'",
        "'b2sigp_style'			  '1'",
        "'b2news_poteq'             '0'",
        "'b2tfhe_no_current'        '1'",
        "'b2trno_pol_anom_scale' '0.0'",
        "'tallies_netcdf' '1'",
        "'b2mndr_na_min' '1.0e10'",
        "'balance_netcdf'  '1'",
        "'balance_average'  '0'",
        "'eirene_ionising_core'  '0'"
    ]

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)






def run_solps(
    chi = None,
    chi_r = None,
    D = None,
    D_r = None,
    iteration = None,
    dna = 1.0,
    hci = 1.0,
    hce = 1.0,
    n_species = 1,
    timeout_hours = 10,
    run_directory = None,
    set_div_transport = False,
    output_directory = None,
    solps_n_timesteps = 1000,
    solps_dt = 1.0E-5,
    n_proc = 1
):
    """
    Evaluates SOLPS for the provided transport profiles and saves the results.

    :param chi: \
        Array of conductivity values corresponding to the the radial positions given in *chi_r*.

    :param chi_r: \
        Array of radial positions corresponding to the conductivity values given in *chi*.

    :param D: \
        Array of diffusivity values corresponding to the the radial positions given in *D_r*.

    :param D_r: \
        Array of radial positions corresponding to the diffusivity values given in *D*.

    :param int iteration: \
        The iteration number corresponding to the requested solps-run, used to name
        directory in which the solps output is stored.

    :param int timeout_hours: \
        The number of hours to wait for the solps run to complete before raising a
        time-out error.
    """

    # Write the SOLPS input files
    write_solps_b2mn_dat(
        run_directory+'b2mn.dat',
        solps_n_timesteps,  # Number of timesteps
        solps_dt,           # Timestep (s)
        cflme=0.3,          # Multiplier to the electron heat flux limit
        cflmi=1.0,          # Multiplier to the ion heat flux limit
        cflmv=0.5,          # Multiplier to the viscous heat flux limit
        cflal=0.5,          # Multilier to the thermo-electric coefficient flux limit
        cflab=0.5,          #  Multiplier to the friction force flux limit
        label='Gaussian process test case'
    )

    write_solps_transport_paramters(
        run_directory+'b2.transport.parameters',
        n_species,  # Number of species - should be in settings!
        dna,
        0,
        0,
        1.0,
        hci,
        hce,
        0,
        0
    )

    write_solps_transport_inputfile(
        run_directory+'b2.transport.inputfile',
        n_species,
        D_r,D,
        chi_r,chi,
        chi_r,chi,
        set_ana_visc_dperp=True,
        no_pflux=True,
        no_div=set_div_transport
    )

    # Go to the SOLPS run directory to prepare execution
    orig_dir = getcwd()
    chdir(run_directory)

    # Check to see if b2fstate and b2fstati are different
    test = filecmp.cmp(run_directory+'b2fstate',run_directory+'b2fstati')

    if test is False:
        # The initial and final states are different, set them to be the same
        copystatefiles = subprocess.Popen('cp '+run_directory+'b2fstate '+run_directory+'b2fstati',stdout=subprocess.PIPE,shell=True)
        copystatefiles.communicate()

    findstr = 'Submitted batch job'

    if n_proc == 1:
        start_run = subprocess.Popen('itmsubmit',stdout=subprocess.PIPE,shell=True)
    if n_proc > 1:
        start_run = subprocess.Popen('itmsubmit -m "-np '+str(n_proc)+'"',stdout=subprocess.PIPE,shell=True)

    start_run_output = start_run.communicate()[0]

    # Find the batch job number
    tmp = str(start_run_output).find(findstr)
    jobstr = str(start_run_output)[tmp+len(findstr)+1:-3]

    logging.info('[solps_interface] Submitted job '+jobstr)

    # set a time-out point
    timeout = time() + 3600*timeout_hours

    # wait for run completion
    uname = subprocess.Popen('whoami',stdout=subprocess.PIPE,shell=True)
    username = uname.communicate()[0]
    username = str(username.rstrip(),'utf-8')

    test = subprocess.Popen('squeue -u '+username,stdout=subprocess.PIPE,shell=True)
    jobqueue = test.communicate()[0]

    jobpos = str(jobqueue).find(jobstr)

    while True:
        if jobpos == -1: # check if the job is still running
            logging.info('[solps_interface] Job '+jobstr+' has finished')
            break
        else:
            logging.debug('[solps_interface] Job '+jobstr+' is in progress...')

        if time() > timeout: # check to see if we've timed out
            #raise TimeoutError('No SOLPS output file was detected within the time-out limit')
            logging.warning('[solps_interface] No SOLPS output file was detected within the time-out limit')
            cancel_solps(jobstr) # TODO - does this function actually break the while loop? if not is a break needed?

        sleep(30) # wait for 30 seconds between checks

        test = subprocess.Popen('squeue -u '+username, stdout=subprocess.PIPE, shell=True)
        jobqueue = test.communicate()[0]
        jobpos = str(jobqueue).find(jobstr)

    # Run this when SOLPS has finished
    # build the path of the solps output file
    output_path = run_directory + 'balance.nc'
    new_output_path = output_directory + 'solps_run_{}.nc'.format(int(iteration))

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


def reset_solps(run_directory, ref_directory):
    """
    Clears the contents of a SOLPS run directory and replaces it with the files from a reference
    case. To be used if the optimiser has executed too many SOLPS runs that the run_directory
    becomes too large.
    """

    orig_dir = getcwd()

    # Go to the run directory
    chdir(run_directory)

    # Clear the contents of the run directory
    test = subprocess.Popen('rm -rf *',stdout=subprocess.PIPE,shell=True)
    jobqueue = test.communicate()[0]

    # Copy the contents of the reference case
    test = subprocess.Popen('cp '+ref_directory+'* .',stdout=subprocess.PIPE,shell=True)
    jobqueue = test.communicate()[0]
    
    chdir(orig_dir)


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
