import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from stelle import IO
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import gridspec
import os
import json

def abp(t,D,Dr,v0):
    return 4*D*t+2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)

def ap(t,Dr,v0):
    return 2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)


with open('stelle_parameters.json',"r") as f:
    details = json.load(f)


nskip=1#per video di tutto usa 200
time_limit=1
initial_skip=0
diam=1
if details['brownian']==0:
    dt=0.001
else:
    dt=0.0001

name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"
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

msd_c=np.zeros(nmax)
msd_c_std=np.zeros(nmax)

traj.reset_index(level='timestep',inplace=True)
times=pd.unique(traj['timestep'])
trajlist = np.array([data[["x","y"]] for _, data in traj.groupby(["timestep"])])
trajlistc = np.array([data[data.type==1][["x","y"]] for _, data in traj.groupby(["timestep"])])
utr, utc = np.triu_indices(len(times),1)
diffs=np.abs(times[utr]-times[utc])

for i, tval in enumerate(frames):
    cond=diffs==tval
    nvals=min(np.sum(cond),tmax//tval)
    msdval=np.sum((trajlist[utr[cond],:,:] - trajlist[utc[cond],:,:] ) ** 2,axis=2)
    msd[i] = np.mean(msdval)
    msd_std[i] = np.std(msdval)/np.sqrt(nvals)

    msdvalc = np.sum((trajlistc[utr[cond], :, :] - trajlistc[utc[cond], :, :]) ** 2, axis=2)
    msd_c[i] = np.mean(msdvalc)
    msd_c_std[i] = np.std(msdvalc) / np.sqrt(nvals)
    #msd[i]=np.mean((trajlist[utr[cond],1]-trajlist[utc[cond],1])**2+(trajlist[utr[cond],2]-trajlist[utc[cond],2])**2)

t=frames*dt
nlast=4
param, pcov = curve_fit(ap, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
perr = np.sqrt(np.diag(pcov))
print(param)
T=1
gt=1
gr=1
Dv=T/gt
Drv=T/gr
v0v=10
paramv=[Drv,v0v]
print(paramv)

nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

ax[0].errorbar(t,msd,yerr=msd_std,c="b",lw=2,ls="",marker="o", label="all")
ax[0].errorbar(t,msd_c,yerr=msd_c_std,c="r",lw=2,ls="",marker="s", label="core")
ax[0].legend(fontsize=20)
#ax[0].plot(t, ap(t, *paramv), linestyle='--',c="b")
#ax[0].plot(t, ap(t, *param), linestyle=':',c="r")
for i in range(nplots):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$msd[\sigma^2]$", fontsize=20)
    ax[i].set_xlabel(r"$t[\tau]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
fig.savefig("../../data/03_analyzed/msd/"+IO.get_name(details)+"_msd.png", bbox_inches="tight",dpi=300)
        