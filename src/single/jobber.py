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
from builder import Single
from builder import logtimer

def job_maker(details):
    
    runtime=1e7
    dumptime=1e5
    restime=1e7
    
    queue="topol_cpuQ"
    input_name="test_input.run"
    job_name="test_job.pbs"
    
    folder="single"

    single=Single(details)

    #Path("../../data/00_external/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    Path(f"../../data/01_raw/{folder}/"+single.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/02_processed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)
    #Path("../../data/03_analyzed/cycles/"+daisy.name).mkdir(parents=True, exist_ok=True)

    dir_name = f"../../data/01_raw/{folder}/" + single.name
    if delet:
        list_dir=os.listdir(dir_name)
        for item in list_dir:
            if item.endswith(".lammpstrj") or item.endswith(".restart") or item.endswith("log"):
                os.remove(os.path.join(dir_name, item))

    logtimer(dumptime, runtime, dir_name + "/" + "logtime.txt")
    
    lammps_input=ds.LammpsLangevinInput(single, runtime=runtime, restime=restime, dumptime=dumptime, timestep=dt)
    
    lammps_config=ds.LammpsDatafile(single)
    
    pbs_par=ds.PBSParams(name=job_name, fin=input_name, queue=queue)

    slurm_par = ds.SLURMParams(name=job_name, nnodes=1, partition=partition, project=project, qos=qos,  fin=input_name, delta=120)
    
    renderer=ds.SimulationRenderer(lammps_config, lammps_input, pbs_par, slurm_par, n_restarts=n_restarts, delta_box=10, suffix='')
    
    renderer.render_sim(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent/"data"/"01_raw"/folder,exist_ok=True)

sigma=15                #diameter of bead in mm. This does not enter in the simulation, as it is automatically set to 1, just makes it easier to define the other parameters
n_restarts=0            #number of times simulation is automatically restarted on cluster
r_conf=0.0/sigma        #radius of confinement in sigma. If 0, there is no confinement
peclet=35/sigma         #peclet number (self propulsion strength) in sigma/s
brownian=0              #0 for Langevin dynamics, 0 for Brownian dynamics
gamma=10000                #determines the friction coefficient gamma (choose big for overdamped dynamics)
Dr=0.8              #defines the rotational diffusion constant
Dt=0.5                #defines the translational diffusion constant
dt=min(0.001,0.1/gamma)                #timestep in s

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
    details = {"peclet":peclet, "r_conf":r_conf, "brownian":brownian, "gamma":gamma, "Dr":Dr, "Dt":Dt, "seed_start": seed}
    
    job_maker(details)