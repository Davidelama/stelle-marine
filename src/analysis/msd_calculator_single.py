import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from single import IO
import json
import pandas as pd

def abp(t,D,Dr,v0):
    return 4*D*t+2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)


with open('single_parameters.json',"r") as f:
    details = json.load(f)


nskip=1#per video di tutto usa 200
time_limit=1
initial_skip=0
diam=1
dt=1e-3

name="../../data/01_raw/single/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"
print(name)
print("Reading data:")

traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','q1','q4'))
traj[["x","y"]]*=diam
traj["theta"]=np.arctan2(traj["q4"],traj["q1"])*2
#plt.hist(traj.theta, range=(-np.pi, np.pi), bins=np.linspace(-np.pi, np.pi, 20))
#plt.show()

tmax=traj.index.max()[1]
nmax=int(np.log2(tmax))+1

frames=2**np.arange(nmax)
msd=np.zeros(nmax)
msd_std=np.zeros(nmax)

traj.reset_index(level='timestep',inplace=True)
times=traj['timestep'].values
trajlist = np.array([data[["x","y"]] for _, data in traj.groupby(["timestep"])])

utr, utc = np.triu_indices(len(times),1)

diffs=np.abs(times[utr]-times[utc])

for i, tval in enumerate(frames):
    cond=diffs==tval
    nvals=min(np.sum(cond),tmax//tval)
    msdval=np.sum((trajlist[utr[cond],:,:] - trajlist[utc[cond],:,:] ) ** 2,axis=2)
    msd[i] = np.mean(msdval)
    msd_std[i] = np.std(msdval)/np.sqrt(nvals)
    #msd[i]=np.mean((trajlist[utr[cond],1]-trajlist[utc[cond],1])**2+(trajlist[utr[cond],2]-trajlist[utc[cond],2])**2)

t=frames*dt
results=pd.DataFrame(columns=["time", "msd", "msd_std"], data=np.c_[t,msd,msd_std])
results.to_csv("../../data/02_processed/msd/"+IO.get_name(details)+"_msd.txt", sep=' ', index=False)
nlast=4
param, pcov = curve_fit(abp, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
perr = np.sqrt(np.diag(pcov))
print(param)
Dv=details["Dt"]#0.01
Drv=details["Dr"]
gt=details["gamma"]
I=2/5*0.5**2
m=1
gr=gt
v0v=details["peclet"]
paramv=[Dv,Drv,v0v]
print(paramv)


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
ax.set_ylabel(r"$msd[\sigma^2]$",fontsize=20)
ax.set_xlabel(r"$t[\tau]$",fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.errorbar(t,msd,yerr=msd_std,c="k",lw=2,ls="",marker="o")
ax.plot(t, abp(t, *paramv), linestyle='--',c="b")
ax.plot(t, abp(t, *param), linestyle=':',c="r")
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

        