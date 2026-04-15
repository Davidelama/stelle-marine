import builder as ds
import numpy as np
from pathlib import Path
import os
from builder import logtimer

def job_maker(details):
    
    runtime = 2e9#2e9#5e6# #cannot be larger than 2e9!
    dumptime = 1e6#1e6#2e4#
    dumptime_msd = 2e7#2e7#5e4#
    dumptime_angle = 1e4#1e4#2e1#
    restime = 1e8
    
    queue="topolCPUQ"
    input_name="test_input.run"
    job_name="test_job.pbs"
    
    folder="stelle"

    dbuild=ds.DaisyBuilder(details)
    daisy=dbuild.build_system()

    #Path("../../data/00_external/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    Path(f"../../data/01_raw/{folder}/"+daisy.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/02_processed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/03_analyzed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    print("Creating: " + daisy.name)
    dir_name = f"../../data/01_raw/{folder}/" + daisy.name
    if delet:
        list_dir=os.listdir(dir_name)
        for item in list_dir:
            #if item.endswith(".lammpstrj") or item.endswith(".restart") or item.endswith("log") or item.endswith("dat")::
            os.remove(os.path.join(dir_name, item))

    logtimer(dumptime_msd, runtime, dir_name + "/" + "logtime.txt")
    
    lammps_input=ds.LammpsLangevinInput(daisy, runtime=runtime, restime=restime, dumptime=dumptime, timestep=dt, dumptime_angle=dumptime_angle, dumptime_msd=dumptime_msd)
    
    lammps_config=ds.LammpsDatafile(daisy)
    
    pbs_par=ds.PBSParams(name=job_name, fin=input_name, queue=queue)

    slurm_par = ds.SLURMParams(name=job_name, nnodes=1, ncores=4, partition=partition, project=project, qos=qos,  fin=input_name, delta=120)
    
    renderer=ds.SimulationRenderer(lammps_config, lammps_input, pbs_par, slurm_par, n_restarts=n_restarts, delta_box=10, suffix='')
    
    renderer.render_sim(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent/"data"/"01_raw"/folder,exist_ok=True)


sigma=15                    #diameter of bead in mm. This does not enter in the simulation, as it is automatically set to 1, just makes it easier to define the other parameters
n_mol=1                     #number of molecules (not yet fully implemented)
n_beads=[11]#[7,9,11,13]#[5,7,9,11,13,15]    #number of beads per arm
r_core=20/(2*sigma)         #core radius (units of bead diameters)
n_restarts=0                #number of times simulation is automatically restarted on cluster
gh=0                        #molecules do not interact with each other (not yet fully implemented)
r_conf=150.0/sigma          #radius of confinement in sigma. If 0, there is no confinement
peclet=35/sigma             #peclet number (self propulsion strength) in sigma/s
r_bond=17/sigma             #size of bead-bead bonds in sigma
r_cbond=(10-1.5)/sigma      #distance of arm grafting locations on the core with respect to the core center in sigma
brownian=0                  #0 for Langevin dynamics, 0 for Brownian dynamics
v0c=35/sigma
Drc=0.8
Dt=0.01 #v0c**2/Drc/2                   #defines translational diffusion coefficient in sigma^2/s
Dr=0.8                      #defines the rotational diffusion coefficient Dr in rad^2/s
gamma=100                   #determines the friction coefficient gamma (choose big for overdamped dynamics)
dt=min(0.001,0.01/gamma)    #timestep in s
if brownian:                #Brownian dynamics requires a smaller timestep
    dt=0.0001
fs=[3,6]#[3,4,5,6]                #number of arms of the star (functionality). This is also an example of vector of parameters for multiple simulations
r_ints=[0]#[4,5,6,7,8,9,10,12,14,16,18,20]          #calculation of effective interaction between stars in sigma. If r_int>0, this automatically sets mol=2, gh=0, rconf=0, fixes the core positions and outputs their forces
contact=1                   #if 1, introduces contact friction between particles and walls
rolling=1                   #if 1, introduce rolling friction in the system
d_pass=[0.05]#[0.01, 0.03,0.05,0.07,0.1,0.15,0.2]                #if >0, introduce passive particles to be captured by the stars
r_pass=5/sigma             #radius of passive particles
Dt_pass=[0.015]               #traslational diffusion of passive particles
gam_pass=[30]                  #friction coefficient of passive particles
kn_pass=[1e5,1e6]
en_pass=[50.0,100]
kt_pass=[1e6,1e7]
er_pass=[0.1,1.0]
mu_pass=[1,10.0]


delet=True              #delete old data
partition='boost_usr_prod'
qos='normal' #'boost_qos_lprod'
project='INF26_biophys'



Nsims=1
if Nsims==1:
    seeds=[None]
else:
    seeds=1+np.arange(Nsims)*1000

#al_defs=np.linspace(0,100,num=26,endpoint=True)

for seed in seeds:
    for fval in fs:
        for nval in n_beads:
            for rint in r_ints:
                for d_pas in d_pass:
                    for Dt_pas in Dt_pass:
                        for gam_pas in gam_pass:
                            for kn_pas in kn_pass:
                                for en_pas in en_pass:
                                    for kt_pas in kt_pass:
                                        for er_pas in er_pass:
                                            for mu_pas in mu_pass:
                                                details = {"n_beads": nval, "n_mol": n_mol,"functionality": fval,
                                                    "r_core": r_core, "peclet":peclet, "r_conf":r_conf, "r_bond":r_bond, "r_cbond":r_cbond, "r_int":rint, "d_pass":d_pas, "r_pass":r_pass, "Dt_pass":Dt_pas, "gam_pass":gam_pas, "kn_pass":kn_pas, "en_pass":en_pas, "kt_pass":kt_pas, "er_pass":er_pas, "mu_pass":mu_pas, "seed_start": seed, "ghost":gh, "brownian":brownian, "Dr":Dr, "Dt":Dt, "gamma":gamma, "contact":contact, "rolling":rolling}

                                                job_maker(details)