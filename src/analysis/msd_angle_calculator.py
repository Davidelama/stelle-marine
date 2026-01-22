import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import gridspec
import sys
import os
import math
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from stelle import IO
import json

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

with open('stelle_parameters.json',"r") as f:
    details = json.load(f)

nskip=1#per video di tutto usa 200
time_limit=1
initial_skip=0
diam=1
if details['brownian']==0:
    dt=min(0.001,0.01/details["gamma"])
else:
    dt=0.0001

name_unwrap="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_angle_unwrap.lammpstrj"

traj_unwrap = IO.reconstruct_traj([name_unwrap], cols=('at_id','mux','muy'))
traj_unwrap["theta"]=np.arctan2(traj_unwrap["muy"],traj_unwrap["mux"])

traj_unwrap.reset_index(level='timestep',inplace=True)
times_unwrap=pd.unique(traj_unwrap['timestep'])
trajlistunwrap = np.array([data["theta"] for _, data in traj_unwrap.groupby(["timestep"])])
#plt.plot(times_unwrap,trajlistunwrap[:,0]/np.pi,'b',ls="--")
trajlistunwrap=np.unwrap(trajlistunwrap,axis=0)
#plt.plot(times_unwrap,trajlistunwrap[:,0]/np.pi,'r',ls="--")

name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"
print(name)
print("Reading data:")

traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','mux','muy'))
traj["theta"]=np.arctan2(traj["muy"],traj["mux"])

tmax=traj.index.max()[1]
nmax=int(np.log2(tmax))+1

frames=2**np.arange(nmax)

msd_c=np.zeros(nmax)
msd_c_std=np.zeros(nmax)

traj.reset_index(level='timestep',inplace=True)
times=pd.unique(traj['timestep'])
unwrap_idx=np.zeros(len(times))
trajlistc = np.array([data[data.type==1]["theta"] for _, data in traj.groupby(["timestep"])])
#plt.plot(times,np.unwrap(trajlistc[:,0])/np.pi,'b',lw=2)
for i, time in enumerate(times):
    idx = find_nearest(times_unwrap,time)
    trajlistc[i]+=np.rint((trajlistunwrap[idx]-trajlistc[i])/(2*np.pi))*np.pi*2

#plt.plot(times,trajlistc[:,0]/np.pi,'r',lw=2)
#plt.show()
plt.plot(times_unwrap[1:],np.abs(trajlistunwrap[1:,0]-trajlistunwrap[:-1,0])/np.pi,'g')
plt.show()
utr, utc = np.triu_indices(len(times),1)
diffs=np.abs(times[utr]-times[utc])


utr, utc = np.triu_indices(len(times),1)
diffs=np.abs(times[utr]-times[utc])

for i, tval in enumerate(frames):
    cond=diffs==tval
    nvals=min(np.sum(cond),tmax//tval)

    msdvalc = (trajlistc[utr[cond], :] - trajlistc[utc[cond], :]) ** 2
    msd_c[i] = np.mean(msdvalc)
    msd_c_std[i] = np.std(msdvalc) / np.sqrt(nvals)
    #msd[i]=np.mean((trajlist[utr[cond],1]-trajlist[utc[cond],1])**2+(trajlist[utr[cond],2]-trajlist[utc[cond],2])**2)

t=frames*dt
results=pd.DataFrame(columns=["time", "angle_msd_core", "angle_msd_core_std"], data=np.c_[t,msd_c,msd_c_std])
results.to_csv("../../data/02_processed/msd/"+IO.get_name(details)+"_angle_msd.txt", sep=' ', index=False)
nlast=4

nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

ax[0].errorbar(t,msd_c,yerr=msd_c_std,c="r",lw=2,ls="",marker="s", label="core")
ax[0].legend(fontsize=20)
for i in range(nplots):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$msd[$rad$^2]$", fontsize=20)
    ax[i].set_xlabel(r"$t[\tau]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
fig.savefig("../../data/03_analyzed/msd/"+IO.get_name(details)+"_angle_msd.png", bbox_inches="tight",dpi=300)
        