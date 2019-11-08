
from os import getcwd, chdir
from time import sleep, time
import subprocess
import filecmp
from scipy.io import netcdf, loadmat
from numpy import arange, min, isfinite, mean, squeeze, sum, argmin, abs, shape, where
from scipy.interpolate import interp1d, interp2d
from copy import deepcopy

import matplotlib.pyplot as plt

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






def run_solps(chi = None, chi_r = None, D = None, D_r = None, iteration = None, timeout_hours = 10, run_directory = None, 
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

    # set solps running here
    n_species = 9

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
                                    15.0,
                                    0,
                                    0,
                                    1.0,
                                    15.0,
                                    15.0,
                                    0,
                                    0)

    write_solps_transport_inputfile(run_directory+'b2.transport.inputfile',
                                    n_species,
                                    D_r,D,
                                    chi_r,chi,
                                    chi_r,chi,
                                    set_ana_visc_dperp=True,
                                    no_pflux=True,
                                    no_div=False)

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

    print('[solps_interface] Submitted job '+jobstr)

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
            print('[solps_interface] Job '+jobstr+' has finished')
            break
        else:
            print('[solps_interface] Job '+jobstr+' is in progress...')

        if time() > timeout: # check to see if we've timed out
            raise TimeoutError('No SOLPS output file was detected within the time-out limit')

        sleep(30) # wait for 30 seconds between checks

        test = subprocess.Popen('squeue -u '+username,stdout=subprocess.PIPE,shell=True)
        jobqueue = test.communicate()[0]
        jobpos = str(jobqueue).find(jobstr)

    # Run this when SOLPS has finished
    # build the path of the solps output file
    output_path = run_directory + 'balance.nc'
    new_output_path = output_directory + 'solps_run_{}.nc'.format(iteration)

    # copy the SOLPS output to the solps_output_directory
    copyoutputfiles = subprocess.Popen('cp '+output_path+' '+new_output_path,stdout=subprocess.PIPE,shell=True)
    copyoutputfiles.communicate()

    chdir(orig_dir)


def read_solps_data(filename = None):
    """
    Reads data from SOLPS output files
    """

    # load solps data
    f = netcdf.netcdf_file(filename,'r')
    crx = deepcopy(f.variables['crx'].data)
    cry = deepcopy(f.variables['cry'].data)
    vol = deepcopy(f.variables['vol'].data)
    solps_ne = deepcopy(f.variables['ne'].data)
    solps_na = deepcopy(f.variables['na'].data)
    solps_species = deepcopy(f.variables['species'].data)
    solps_te = deepcopy(f.variables['te'].data)/1.602E-19
    solps_ti = deepcopy(f.variables['ti'].data)/1.602E-19
    solps_vi = deepcopy(f.variables['ua'].data)
    solps_dab2 = deepcopy(f.variables['dab2'].data[:,:,0:-2])   # Atomic density
    solps_dmb2 = deepcopy(f.variables['dmb2'].data[:,:,0:-2])   # Molecular density
    eirene_sion = sum(f.variables['eirene_mc_papl_sna_bal'].data,axis=0)[1,:,:]
    eirene_eloss = sum(f.variables['eirene_mc_eael_she_bal'].data,axis=0)
    b2_ploss = deepcopy(f.variables['b2stel_she_bal'].data)

    # Estimate the total radiated power emissivity
    prad = b2_ploss/vol+(eirene_eloss-13.6*eirene_sion*1.602E-19)/vol

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
    output['sep_indx'] = sep
    output['omp_indx'] = omp
    output['cen_x']    = solps_cen_x
    output['cen_y']    = solps_cen_y
    
    return output



def evaluate_log_posterior(iteration = None, directory = None, diagnostic_data_file = None,
                           diagnostic_data_desc = None):
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

    indx = diagnostic_data_desc.index('Data Time')
    data_time = float(diagnostic_data_file[indx])

    # Load the Thomson scattering data
    indx = diagnostic_data_desc.index('Midplane TS')
    ts_data = loadmat(directory+diagnostic_data_file[indx])

    ts_r_shift = 0.0
    ts_z_shift = 0.0
    
    if 'TS R shift' in diagnostic_data_desc:
        indx = diagnostic_data_desc.index('TS R shift')
        ts_r_shift = float(diagnostic_data_file[indx])

    if 'TS Z shift' in diagnostic_data_desc:
        indx = diagnostic_data_desc.index('TS Z shift')
        ts_z_shift = float(diagnostic_data_file[indx])

    ts_t    = squeeze(ts_data['ts'][0][0][12])
    ts_ne   = ts_data['ts'][0][0][4]
    ts_te   = ts_data['ts'][0][0][2]
    ts_ene  = ts_data['ts'][0][0][5]
    ts_ete  = ts_data['ts'][0][0][3]
    ts_r    = squeeze(ts_data['ts'][0][0][8])+ts_r_shift
    ts_z    = squeeze(ts_data['ts'][0][0][9])+ts_z_shift
    ts_rho  = ts_data['ts'][0][0][10]
    ts_psi  = ts_data['ts'][0][0][11]
    
    ts_tindx = argmin(abs(data_time-ts_t))

    # Read the equilibrium of the SOLPS grid
    indx = diagnostic_data_desc.index('Grid Equilibrium')
    geq = loadmat(directory+diagnostic_data_file[indx])
    indx = diagnostic_data_desc.index('Grid Time')
    grid_time = float(diagnostic_data_file[indx])

    gpsi_r = squeeze(geq['equil']['R'][0][0])
    gpsi_z = squeeze(geq['equil']['Z'][0][0])
    gpsi_t = squeeze(geq['equil']['time'][0][0])
    gpsi_n = squeeze(geq['equil']['psi_n'][0][0])
    gpsi_raxis = squeeze(geq['equil']['r_axis'][0][0])
    gpsi_zaxis = squeeze(geq['equil']['z_axis'][0][0])
    gpsi_tindx = argmin(abs(gpsi_t-grid_time))

    gpsin = gpsi_n[:,:,gpsi_tindx]
    gpsin_intrp = interp2d(gpsi_r,gpsi_z,gpsin.transpose())

    solps_grid_psin = solps_cen_x*0.0
    for i in arange(shape(solps_cen_x)[0]):
        for j in arange(shape(solps_cen_x)[1]):
            solps_grid_psin[i,j] = gpsin_intrp(solps_data['cen_x'][i,j],solps_data['cen_y'][i,j])

    # Read the equilibrium of the data
    indx = diagnostic_data_desc.index('Equilibrium')
    eq = loadmat(directory+diagnostic_data_file[indx])

    psi_r = squeeze(eq['equil']['R'][0][0])
    psi_z = squeeze(eq['equil']['Z'][0][0])
    psi_t = squeeze(eq['equil']['time'][0][0])
    psi_n = squeeze(eq['equil']['psi_n'][0][0])
    psi_tindx = argmin(abs(psi_t-data_time))

    psin = psi_n[:,:,psi_tindx]
    psin_intrp = interp2d(psi_r,psi_z,psin.transpose())

    # Interpolate the equilibrium normalised flux to the Thomson scattering points
    solps_ts_psin = ts_r*0.0
    for i in arange(len(ts_r)):
        solps_ts_psin[i] = gpsin_intrp(ts_r[i],ts_z[i])

    # Interpolate the normalised flux from the shot the data was recorded
    data_ts_psin = ts_ne[ts_tindx,:]*0.0
    for i in arange(len(data_ts_psin)):
        data_ts_psin[i] = psin_intrp(ts_r[i],ts_z[i])

    # HACK!! Only take the lower half of the profile
    ts_zindx = where(ts_z < 0.0)[0]

    # Collect together the arrays needed to calculate chi square
    thomson_psin = squeeze(data_ts_psin[ts_zindx])
    thomson_ne   = squeeze(ts_ne[ts_tindx,ts_zindx])
    thomson_ene  = squeeze(ts_ene[ts_tindx,ts_zindx])
    thomson_te   = squeeze(ts_te[ts_tindx,ts_zindx])
    thomson_ete  = squeeze(ts_ete[ts_tindx,ts_zindx])

    omp = solps_data['omp_indx']

    solps_psin_mp_profile = squeeze(solps_grid_psin[:,omp])
    solps_ne_mp_profile   = squeeze(solps_data['ne'][:,omp])
    solps_te_mp_profile   = squeeze(solps_data['te'][:,omp])

    # Calculate log-probability for the Thomson data
    interpne = interp1d(solps_psin_mp_profile,solps_ne_mp_profile)
    mp_density_logprob = 0.0
    for i in arange(len(thomson_ne)):
        if thomson_psin[i] > min(solps_psin_mp_profile) and isfinite(thomson_ne[i]):
            mp_density_logprob = mp_density_logprob+(thomson_ne[i]-interpne(thomson_psin[i]))**2/thomson_ene[i]**2
    mp_density_logprob *= -0.5

    interpte = interp1d(solps_psin_mp_profile,solps_te_mp_profile)
    mp_e_temperature_logprob = 0.0
    for i in arange(len(thomson_te)):
        if thomson_psin[i] > min(solps_psin_mp_profile) and isfinite(thomson_te[i]):
            mp_e_temperature_logprob = mp_e_temperature_logprob+(thomson_te[i]-interpte(thomson_psin[i]))**2/thomson_ete[i]**2
    mp_e_temperature_logprob *= -0.5

    #plt.figure()
    #plt.errorbar(thomson_psin,thomson_te,thomson_ete,fmt='.b')
    #plt.hold(True)
    #plt.plot(solps_psin_mp_profile,solps_te_mp_profile,'.r')
    #plt.hold(False)
    #plt.show()

    print(['Midplane ne profile chi square ', mp_density_logprob])
    print(['Midplane Te profile chi square' , mp_e_temperature_logprob])

    # calculate the log-posterior probability
    log_posterior = mp_density_logprob + mp_e_temperature_logprob

    return log_posterior # return the log-posterior
