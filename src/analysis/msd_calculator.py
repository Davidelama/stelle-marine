import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import gridspec
import sys
import os
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from stelle import IO
import json
from stelle.stelle_analysis import msd_func

def abp(t,D,Dr,v0):
    return 4*D*t+2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)

def ap(t,Dr,v0):
    return 2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)

def lp(t,D,gm):
    return 2*D/gm*(t*gm+np.exp(-t*gm)-1)    

def alp(t,D,Dr,v0,gm):
    alpha=Dr/gm #big means big inertia
    return 4/gm*(D+v0**2/Dr*1/(1-1/alpha**2))*(t*gm+np.exp(-t*gm)-1)+2*v0**2/Dr**2*1/(1-alpha**2)*(Dr*t+np.exp(-Dr*t)-1)


with open('stelle_parameters.json',"r") as f:
    details = json.load(f)


name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"

traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','mux','muy'))
results = msd_func(traj,details)
results.to_csv("../../data/02_processed/msd/"+IO.get_name(details)+"_msd.txt", sep=' ', index=False)

t=results["time"]
msd=results["msd_tot"]
msd_std = results["msd_tot_std"]
msd_c = results["msd_core"]
msd_c_std = results["msd_core_std"]

nlast=4
#param, pcov = curve_fit(abp, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
param, pcov = curve_fit(ap, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
perr = np.sqrt(np.diag(pcov))
print(param)
Dv=details["Dt"]#0.01
Drv=details["Dr"]
gt=details["gamma"]
I=2/5*0.5**2
m=1
gr=gt
v0v=details["peclet"]
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
ax[0].plot(t, ap(t, *paramv), linestyle='--',c="b")
ax[0].plot(t, ap(t, *param), linestyle=':',c="r")
#ax[0].plot(t, abp(t, *param), linestyle=':',c="r")
for i in range(nplots):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$msd[\sigma^2]$", fontsize=20)
    ax[i].set_xlabel(r"$t[\tau]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
fig.savefig("../../data/03_analyzed/msd/"+IO.get_name(details)+"_msd.png", bbox_inches="tight",dpi=300)
        