
from os import getcwd, chdir
from os.path import exists
from time import sleep, time
import logging
import subprocess
import filecmp
from scipy.io import netcdf
from numpy import array, arange, isfinite, mean, sum, where, zeros, sqrt, vstack, unique, log
from copy import deepcopy
import matplotlib.path as mplPath
from pandas import read_hdf

def write_solps_transport_inputfile(filename,
                                    n_species,
                                    grid_dperp,values_dperp,
                                    grid_chieperp,values_chieperp,
                                    grid_chiiperp,values_chiiperp,
                                    set_ana_visc_dperp=True,
                                    no_pflux=True,
                                    no_div=False):

    '''
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
    '''

    # Define list to write the file lines to
    sout = []
    sout.append(' &TRANSPORT')

    # Write the anomalous radial particle diffusivity profile
    sout.append(' ndata(1, 1 , 1 )= '+str(len(grid_dperp))+' ,')
    for i in range(len(grid_dperp)):
        str1 = "{0:2.0f}".format(i+1)
        str2 = "{0:6.3f}".format(grid_dperp[i])
        str3 = "{0:6.3f}".format(values_dperp[i])
        sout.append(' tdata(1,'+str1 +' , 1 , 1 )= '+str2+' , tdata(2,'+str1+' , 1 , 1 )= '+str3+' ,')

    # If requested, write the anomalous viscosity
    if set_ana_visc_dperp:
        sout.append(' ndata(1, 7 , 1 )= '+str(len(grid_dperp))+' ,')
        for i in range(len(grid_dperp)):
            str1 = "{0:2.0f}".format(i+1)
            str2 = "{0:6.3f}".format(grid_dperp[i])
            str3 = "{0:6.3f}".format(values_dperp[i])
            sout.append(' tdata(1,'+str1 +' , 7 , 1 )= '+str2+' , tdata(2,'+str1+' , 7 , 1 )= '+str3+' ,')

    # Write the anomalous radial electron heat diffusivity profile
    sout.append(' ndata(1, 3 , 1 )= '+str(len(grid_chiiperp))+' ,')
    for i in range(len(grid_chieperp)):
        str1 = "{0:2.0f}".format(i+1)
        str2 = "{0:6.3f}".format(grid_chieperp[i])
        str3 = "{0:6.3f}".format(values_chieperp[i])
        sout.append(' tdata(1,'+str1 +' , 3 , 1 )= '+str2+' , tdata(2,'+str1+' , 3 , 1 )= '+str3+' ,')

    # Write the anomalous radial ion heat diffusivity profile
    sout.append(' ndata(1, 4 , 1 )= '+str(len(grid_chiiperp))+' ,')
    for i in range(len(grid_chiiperp)):
        str1 = "{0:2.0f}".format(i+1)
        str2 = "{0:6.3f}".format(grid_chiiperp[i])
        str3 = "{0:6.3f}".format(values_chiiperp[i])
        sout.append(' tdata(1,'+str1 +' , 4 , 1 )= '+str2+' , tdata(2,'+str1+' , 4 , 1 )= '+str3+' ,')

    # Need to write addspec information for other species...
    #if n_species > 2:
    #    for i in np.arange(3,n_species+1):
    #        str1 = "{0:2.0f}".format(i)
    #        sout.append(' addspec('+str1+',1,1)='+str1+' ,')
    #        sout.append(' addspec('+str1+',3,1)='+str1+' ,')

    sout.append(' addspec(3,1,1)=3 ,')
    sout.append(' addspec(3,3,1)=3 ,')
    sout.append(' addspec(4,1,1)=4 ,')
    sout.append(' addspec(4,3,1)=4 ,')
    sout.append(' addspec(5,1,1)=5 ,')
    sout.append(' addspec(5,3,1)=5 ,')
    sout.append(' addspec(6,1,1)=6 ,')
    sout.append(' addspec(6,3,1)=6 ,')
    sout.append(' addspec(7,1,1)=7 ,')
    sout.append(' addspec(7,3,1)=7 ,')
    sout.append(' addspec(8,1,1)=8 ,')
    sout.append(' addspec(8,3,1)=8 ,')
    sout.append(' addspec(10,1,1)=10 ,')
    sout.append(' addspec(10,3,1)=10 ,')
    sout.append(' addspec(11,1,1)=11 ,')
    sout.append(' addspec(11,3,1)=11 ,')
    sout.append(' addspec(12,1,1)=12 ,')
    sout.append(' addspec(12,3,1)=12 ,')
    sout.append(' addspec(13,1,1)=13 ,')
    sout.append(' addspec(13,3,1)=13 ,')
    sout.append(' addspec(14,1,1)=14 ,')
    sout.append(' addspec(14,3,1)=14 ,')
    sout.append(' addspec(15,1,1)=15 ,')
    sout.append(' addspec(15,3,1)=15 ,')
    sout.append(' addspec(16,1,1)=16 ,')
    sout.append(' addspec(16,3,1)=16 ,')

    if no_pflux == True:
        sout.append(' no_pflux=.true. ')

    if no_div == True:
        sout.append(' no_div=.true. ')        

    sout.append(' /')

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)   






def write_solps_transport_paramters(filename,
                                    n_species,
                                    dna,
                                    dpa,
                                    vla,
                                    vsa,
                                    hci,
                                    hce,
                                    sig,
                                    alf):

    '''
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
    '''

    # Prepare the file contents in a list
    sout = []
    sout.append(' &transport')
    sout.append(' flag_dna=1, parm_dna='+str(int(n_species))+'*'+str(dna)+',')
    sout.append(' flag_dpa=1, parm_dpa='+str(int(n_species))+'*'+str(dpa)+',')
    sout.append(' flag_vla=1, parm_vla='+str(int(n_species))+'*'+str(vla)+',')
    sout.append(' flag_vsa=1, parm_vsa='+str(int(n_species))+'*'+str(vsa)+',')
    sout.append(' flag_hci=1, parm_hci='+str(int(n_species))+'*'+str(hci)+',')
    sout.append(' flag_hce=1, parm_hce='+str(hce)+',')
    sout.append(' flag_sig=1, parm_sig='+str(sig)+',')
    sout.append(' flag_alf=1, parm_alf='+str(alf)+',')
    sout.append(' /')

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)






def write_solps_b2mn_dat(filename,
                         n_timestep,
                         d_timestep,
                         cflme=0.3, # Multiplier to the electron heat flux limit
                         cflmi=1.0, # Multiplier to the ion heat flux limit
                         cflmv=0.5, # Multiplier to the viscous heat flux limit
                         cflal=0.5, # Multilier to the thermo-electric coefficient flux limit
                         cflab=0.5, #  Multiplier to the friction force flux limit
                         label='Gaussian process test case'
                         ):

    sout = []

    sout.append('*label          (lblmn: character*60)')
    sout.append("  '"+label+"'")
    sout.append('*b2cmpa  basic parameters')
    sout.append('*b2cmpb  boundary conditions')
    sout.append('*b2cmpt  transport coefficients')
    sout.append('*cflim    cflme    cflmi    cflmv     cflal     cflab')
    sout.append("  '-1'        "+str(cflme)+"      "+str(cflmi)+"      "+str(cflmv)+"      "+str(cflal)+"      "+str(cflab)+"        0        0        0")
    sout.append('*endphy')
    sout.append("'b2mndr_ntim'     '"+str(n_timestep)+"'")
    sout.append("'b2mndr_dtim'     '"+str(d_timestep)+"'")
    sout.append("'b2mndt_nstg0'    '1'")
    sout.append("'b2mndt_nstg1'    '1'")
    sout.append("'b2mndt_nstg2'    '10'")
    sout.append("'b2mndt_rxf'      '0.5'")
    sout.append("'b2mndr_cpu'      '80000.0'")
    sout.append("'b2mndr_savecpu'  '1000.0'")
    sout.append("'b2mndr_stim'     '-5'")
    sout.append("'b2mndr_b2time'   '10'")
    sout.append("'b2stbc_boundary_namelist' '1'")
    sout.append("'b2stbr_neutrals_namelist' '1'")
    sout.append("'b2tqna_transport_namelist' '1'")
    sout.append("'b2tqna_inputfile' '1'")
    sout.append("'b2tqna_ballooning' '0'")
    sout.append("'b2tqna_ballooning_rescale' '1'")
    sout.append("'b2mndr_eirene'                   '1'")
    sout.append("'b2mndr_rescale_neutrals_sources' '1e-20'")
    sout.append("'b2sigp_style'			  '1'")
    sout.append("'b2news_poteq'             '0'")
    sout.append("'b2tfhe_no_current'        '1'")
    sout.append("'b2trno_pol_anom_scale' '0.0'")
    sout.append("'tallies_netcdf' '1'")
    sout.append("'b2mndr_na_min' '1.0e10'")
    sout.append("'balance_netcdf'  '1'")
    sout.append("'balance_average'  '0'")
    #sout.append("'eirene_ionising_core'  '1'")
    sout.append("'eirene_ionising_core'  '0'")

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)






def run_solps(chi = None, chi_r = None, D = None, D_r = None, iteration = None, 
              dna = 1.0, hci = 1.0, hce = 1.0, n_species = 1,
              timeout_hours = 10, run_directory = None, set_div_transport = False,
              output_directory = None, solps_n_timesteps = 1000, solps_dt = 1.0E-5, n_proc = 1):
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
    write_solps_b2mn_dat(run_directory+'b2mn.dat',
                         solps_n_timesteps,      # Number of timesteps
                         solps_dt,               # Timestep (s)
                         cflme=0.3,              # Multiplier to the electron heat flux limit
                         cflmi=1.0,              # Multiplier to the ion heat flux limit
                         cflmv=0.5,              # Multiplier to the viscous heat flux limit
                         cflal=0.5,              # Multilier to the thermo-electric coefficient flux limit
                         cflab=0.5,              #  Multiplier to the friction force flux limit
                         label='Gaussian process test case')

    write_solps_transport_paramters(run_directory+'b2.transport.parameters',
                                    n_species, # Number of species - should be in settings!
                                    dna,
                                    0,
                                    0,
                                    1.0,
                                    hci,
                                    hce,
                                    0,
                                    0)

    write_solps_transport_inputfile(run_directory+'b2.transport.inputfile',
                                    n_species,
                                    D_r,D,
                                    chi_r,chi,
                                    chi_r,chi,
                                    set_ana_visc_dperp=True,
                                    no_pflux=True,
                                    no_div=set_div_transport)

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

        test = subprocess.Popen('squeue -u '+username,stdout=subprocess.PIPE,shell=True)
        jobqueue = test.communicate()[0]
        jobpos = str(jobqueue).find(jobstr)

    # Run this when SOLPS has finished
    # build the path of the solps output file
    output_path = run_directory + 'balance.nc'
    new_output_path = output_directory + 'solps_run_{}.nc'.format(iteration)

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
        new_output_path = output_directory + 'solps_input_files_{}.zip'.format(iteration)
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


def read_solps_data(filename = None):
    """
    Reads data from SOLPS output files
    """

    # Physical constants
    el_ch = 1.602E-19
    me = 9.109E-31
    mi = 3.343E-27  # Deuterium mass!

    # load solps data
    f = netcdf.netcdf_file(filename,'r')
    crx = deepcopy(f.variables['crx'].data)
    cry = deepcopy(f.variables['cry'].data)
    vol = deepcopy(f.variables['vol'].data)
    solps_ne = deepcopy(f.variables['ne'].data)
    solps_na = deepcopy(f.variables['na'].data)
    solps_species = deepcopy(f.variables['species'].data)
    solps_te = deepcopy(f.variables['te'].data)/el_ch
    solps_ti = deepcopy(f.variables['ti'].data)/el_ch
    solps_vi = deepcopy(f.variables['ua'].data)
    solps_dab2 = deepcopy(f.variables['dab2'].data[:,:,0:-2])   # Atomic density
    solps_dmb2 = deepcopy(f.variables['dmb2'].data[:,:,0:-2])   # Molecular density
    eirene_sion = sum(f.variables['eirene_mc_papl_sna_bal'].data,axis=0)[1,:,:]
    eirene_eloss = sum(f.variables['eirene_mc_eael_she_bal'].data,axis=0)
    b2_ploss = deepcopy(f.variables['b2stel_she_bal'].data)

    # Estimate the total radiated power emissivity
    prad = b2_ploss/vol+(eirene_eloss-13.6*eirene_sion*el_ch)/vol

    # Estimate the ion saturation current density
    jsat = el_ch*solps_ne*sqrt(el_ch*(solps_te+solps_ti)/(me+mi))

    # Estimate the positions of the the separatrix and outer mid-plane
    sep = deepcopy(f.variables['jsep'].data)+2 # Separatrix ring
    omp = deepcopy(f.variables['jxa'].data)+2 # Outer mid-plane cell index

    # Close the file
    f.close()
    f.flush()

    # Calculate the centres of the SOLPS grid cells
    solps_cen_x = mean(crx,0)
    solps_cen_y = mean(cry,0)

    # Populate the output list
    output = {}
    output['vertex_x'] = crx
    output['vertex_y'] = cry
    output['ne']       = solps_ne
    output['na']       = solps_na
    output['species']  = solps_species
    output['te']       = solps_te
    output['ti']       = solps_ti
    output['vi']       = solps_vi
    output['natm']     = solps_dab2
    output['nmol']     = solps_dmb2
    output['sion']     = eirene_sion
    output['prad']     = prad
    output['jsat']     = jsat
    output['sep_indx'] = sep
    output['omp_indx'] = omp
    output['cen_x']    = solps_cen_x
    output['cen_y']    = solps_cen_y
    
    return output

def calc_point_geo_matrix(samples_r,samples_z,samples_n,cells):
    '''
    Calculates a geometry matrix for pointwise measurements.  Each measurement
    is assumed to comprise of a point cloud of coordinates (r,z).  The position
    of each coordinate within each modelling mesh cell is calculated and returned
    in the geometry matrix

    Inputs:
        - samples_r: radial coordinates
        - samples_z: vertical coordinates
        - samples_n: measurement index {0, ..., number_of_measurements}
        - cells: matplotlib PathCollection containing the modelling grid cells
    '''

    geomat = zeros((int(samples_n.max())+1,len(cells)))
    pts = vstack((samples_r,samples_z)).transpose()

    for i in arange(len(cells)):
        tmp1 = cells[i].contains_points(pts)
        if tmp1.sum() > 0:
            # Found some Thomson scattering points within grid cells
            tmp2 = unique(tmp1*samples_n,return_counts=True)

            for j in arange(len(tmp2[0])):
                if tmp2[0][j] > 0:
                    geomat[int(tmp2[0][j]),i] += tmp2[1][j]

    # Normalise the geometry matrix by the number of samples
    n_samples = unique(samples_n, return_counts=True)

    for i in arange(int(samples_n.max())+1):
        geomat[i,:] /= n_samples[1][i]

    return geomat



def gaussian_likelihood(data, errors, prediction):
    z = (data-prediction)/errors
    return -0.5*(z**2).sum()



def cauchy_likelihood(data, errors, prediction):
    z = (data-prediction)/errors
    return -log(1 + z**2).sum()



def evaluate_log_posterior(iteration = None, directory = None, diagnostic_data_files = None,
                           diagnostic_data_observables = None):
    """
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
    solps_path = directory + 'solps_run_{}.nc'.format(iteration)

    # read the SOLPS data
    solps_data = read_solps_data(solps_path)

    # Create an array of SOLPS cells
    cells = []
    solps_ne_cell = []
    solps_te_cell = []
    solps_ti_cell = []
    solps_jsat_cell = []
    solps_prad_cell = []

    for i in arange(solps_data['vertex_x'].shape[1]):
        for j in arange(solps_data['vertex_x'].shape[2]):
            cells.append(mplPath.Path([[solps_data['vertex_x'][3,i,j],solps_data['vertex_y'][3,i,j]],[solps_data['vertex_x'][1,i,j],solps_data['vertex_y'][1,i,j]],[solps_data['vertex_x'][0,i,j],solps_data['vertex_y'][0,i,j]],[solps_data['vertex_x'][2,i,j],solps_data['vertex_y'][2,i,j]],[solps_data['vertex_x'][3,i,j],solps_data['vertex_y'][3,i,j]]]))
            solps_ne_cell.append(solps_data['ne'][i,j])
            solps_te_cell.append(solps_data['te'][i,j])
            solps_ti_cell.append(solps_data['ti'][i,j])
            solps_jsat_cell.append(solps_data['jsat'][i,j])
            solps_prad_cell.append(solps_data['prad'][i,j])

    solps_ne_cell = array(solps_ne_cell)
    solps_te_cell = array(solps_te_cell)
    solps_ti_cell = array(solps_ti_cell)
    solps_jsat_cell = array(solps_jsat_cell)
    solps_prad_cell = array(solps_prad_cell)

    # storage for the log-probabilities
    gauss_logprobs = []
    cauchy_logprobs = []

    # Read the experimental data
    for filename, observables in zip(diagnostic_data_files, diagnostic_data_observables):

        data = read_hdf(directory+filename+'.h5','data',mode='r')
        data_validindx = isfinite(data['r'])

        # Read the geometry information
        data_r = data['r'].values[data_validindx]
        data_z = data['z'].values[data_validindx]
        sample_r = data['sample_r'].values
        sample_z = data['sample_z'].values
        sample_n = data['sample_n'].values
        data_type = data['measurement_type'].values[0]

        # Check if a geometry matrix has been calculated

        # If not, calculate the geometry matrix
        if data_type == 'point':
            geomat = calc_point_geo_matrix(sample_r,sample_z,sample_n,cells)

        if data_type == 'line':
            raise Exception('Measurement type not yet supported.')

        for tag in observables:
            if tag == 'ne':
                predicted_data = geomat.dot(solps_ne_cell)

            if tag == 'te':
                predicted_data = geomat.dot(solps_te_cell)

            if tag == 'ne_weighted_te':
                predicted_data = geomat.dot(solps_te_cell*solps_ne_cell)/geomat.dot(solps_ne_cell)

            if tag == 'ti':
                predicted_data = geomat.dot(solps_ti_cell)

            if tag == 'jsat':
                predicted_data = geomat.dot(solps_jsat_cell)

            if tag == 'prad':
                predicted_data = geomat.dot(solps_prad_cell)

            # Filter the projected data so that only measurements made within the SOLPS grid
            # are retained
            geo_validindx = where(sum(geomat,axis=1) > 0.99)[0]

            expt_data = data[tag].values[data_validindx][geo_validindx]
            expt_err = data[tag].values[data_validindx][geo_validindx]
            predicted_data = predicted_data[geo_validindx]

            # get indices where both the data and the errors are finite
            finite = where(isfinite(expt_data) & isfinite(expt_err))

            cauchy_logprobs.append( cauchy_likelihood( expt_data[finite], expt_err[finite], predicted_data[finite] ) )
            gauss_logprobs.append( gaussian_likelihood( expt_data[finite], expt_err[finite], predicted_data[finite] ) )

    # calculate the log-posterior probability
    total_cauchy_logprob = sum(cauchy_logprobs)
    total_gauss_logprob = sum(gauss_logprobs)
    return total_gauss_logprob, total_cauchy_logprob






def build_solps_mesh(solps_file):
    from mesh_tools.mesh import Triangle, TriangularMesh

    # load solps data
    f = netcdf.netcdf_file(solps_file,'r')
    crx = deepcopy(f.variables['crx'].data)
    cry = deepcopy(f.variables['cry'].data)

    # Close the file
    f.close()
    f.flush()

    # Calculate the centres of the SOLPS grid cells
    R = mean(crx,axis=0)
    z = mean(cry,axis=0)

    # build a mesh by splitting SOLPS grid cells
    triangles = []
    for i in range(R.shape[0]-1):
        for j in range(R.shape[1]-1):
            p1 = (R[i,j], z[i,j])
            p2 = (R[i+1,j], z[i+1,j])
            p3 = (R[i,j+1], z[i,j+1])
            p4 = (R[i+1,j+1], z[i+1,j+1])
            triangles.append( Triangle(p1, p2, p3) )
            triangles.append( Triangle(p2, p3, p4) )

    return TriangularMesh(triangles = triangles)






def evaluate_midas_posterior(midas_input_file, solps_file):
    posterior = build_midas_posterior(midas_input_file)
    theta = build_midas_parameters(posterior, solps_file)
    return posterior(theta)






def build_midas_posterior(midas_input_file):
    from runpy import run_path
    from os.path import isfile

    if isfile(midas_input_file):  # check to see if the given path is valid
        settings = run_path(midas_input_file)  # run the settings module
    else:
        raise ValueError('{} is not a valid path to an input module'.format(midas_input_file))

    if 'posterior' not in settings:
        raise ValueError('"posterior" was not found in the input module')

    return settings['posterior']






def build_midas_parameters(posterior, solps_file):
    # get all the fields required by the posterior
    field_tags = [ tag for tag in posterior.plasma.parameters.keys() ]

    # load solps data
    f = netcdf.netcdf_file(solps_file,'r')
    R = deepcopy(f.variables['crx'].data)
    z = deepcopy(f.variables['cry'].data)
    solps_ne = deepcopy(f.variables['ne'].data)
    solps_te = deepcopy(f.variables['te'].data)

    # Close the file
    f.close()
    f.flush()

    # Calculate the centres of the SOLPS grid cells
    R = mean(R,axis=0)
    z = mean(z,axis=0)

    # build a map from the vertex coords to the grid indices
    map = {}
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            map[(R[i,j],z[i,j])] = (i,j)

    # get a list of the grid indices for each vertex in order
    vertex_order = [map[v] for v in posterior.plasma.mesh.vertices]

    # prepare the field vectors
    mesh_fields = {
        'Te' : array([solps_te[i,j] for i,j in vertex_order]),
        'ne' : array([solps_ne[i,j] for i,j in vertex_order])
    }

    # construct the parameter vector
    theta = zeros(posterior.plasma.N_params)
    for tag in field_tags:
        theta[posterior.plasma.slices[tag]] = mesh_fields[tag]
