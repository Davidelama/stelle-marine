import builder as ds
import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
from IO import get_name
from copy import copy, deepcopy
from builder import logtimer

def job_maker(details):
    
    runtime=1e7
    dumptime=1e5
    if details["functionality"]*details["n_beads"] >= 1e4:
        restime=1e6
    else:
        restime=1e7
    
    queue="topol_cpuQ"
    input_name="test_input.run"
    job_name="test_job.pbs"
    
    folder="stelle"

    dbuild=ds.DaisyBuilder(details)
    daisy=dbuild.build_system()

    #Path("../../data/00_external/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    Path(f"../../data/01_raw/{folder}/"+daisy.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/02_processed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/03_analyzed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)

    dir_name = f"../../data/01_raw/{folder}/" + daisy.name
    if delet:
        list_dir=os.listdir(dir_name)
        for item in list_dir:
            if item.endswith(".lammpstrj") or item.endswith(".restart") or item.endswith("log"):
                os.remove(os.path.join(dir_name, item))

    logtimer(dumptime, runtime, dir_name + "/" + "logtime.txt")
    
    lammps_input=ds.LammpsLangevinInput(daisy, runtime=runtime, restime=restime, dumptime=dumptime, timestep=dt)
    
    lammps_config=ds.LammpsDatafile(daisy)
    
    pbs_par=ds.PBSParams(name=job_name, fin=input_name, queue=queue)

    slurm_par = ds.SLURMParams(name=job_name, nnodes=1, partition=partition, project=project, qos=qos,  fin=input_name, delta=120)
    
    renderer=ds.SimulationRenderer(lammps_config, lammps_input, pbs_par, slurm_par, n_restarts=n_restarts, delta_box=10, suffix='')
    
    renderer.render_sim(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent/"data"/"01_raw"/folder,exist_ok=True)


sigma=15                #diameter of bead in mm. This does not enter in the simulation, as it is automatically set to 1, just makes it easier to define the other parameters
n_mol=1                 #number of molecules (not yet fully implemented)
n_beads=13              #number of beads per arm
r_core=20/(2*sigma)     #core radius (units of bead diameters)
n_restarts=0            #number of times simulation is automatically restarted on cluster
gh=1                    #molecules do not interact with each other (not yet fully implemented)
r_conf=150.0/sigma      #radius of confinement in sigma. If 0, there is no confinement
peclet=35/sigma         #peclet number (self propulsion strength) in sigma/s
r_bond=17/sigma         #size of bead-bead bonds in sigma
r_cbond=(10-1.5)/sigma  #distance of arm grafting locations on the core with respect to the core center in sigma
brownian=1              #0 for Langevin dynamics, 0 for Brownian dynamics
Dr=0.8                  #if brownian=1, defines the rotational diffusion coefficient Dr in rad^2/s
dt=0.001                #timestep in s
if brownian:            #Brownian dynamics requires a smaller timestep
    dt=0.0001
fs=[6]                  #number of arms of the star (functionality). This is also an example of vector of parameters for multiple simulations



delet=True              #delete old data
partition='boost_usr_prod'
qos='normal'
project='INF25_biophys'



Nsims=1
if Nsims==1:
    seeds=[None]
else:
    seeds=1+np.arange(Nsims)*1000

#al_defs=np.linspace(0,100,num=26,endpoint=True)

for seed in seeds:
    for fval in fs:
        details = {"n_beads": n_beads, "n_mol": n_mol,"functionality": fval,
                "r_core": r_core, "peclet":peclet, "r_conf":r_conf, "r_bond":r_bond, "r_cbond":r_cbond, "seed_start": seed, "ghost":gh, "brownian":brownian, "Dr":Dr}
    
        job_maker(details)