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
from stelle.stelle_analysis import msd_pass_func

def abp(t,D,Dr,v0):
    return 4*D*t+2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)

def ap(t,Dr,v0):
    return 2*v0**2/Dr**2*(Dr*t+np.exp(-t*Dr)-1)

def lp(t,D,gm):
    return 4/gm*(D)*(t*gm+np.exp(-t*gm)-1)

def alp(t,D,Dr,v0,gm):
    alpha=Dr/gm #big means big inertia
    return 4/gm*(D+v0**2/Dr*1/(1-1/alpha**2))*(t*gm+np.exp(-t*gm)-1)+2*v0**2/Dr**2*1/(1-alpha**2)*(Dr*t+np.exp(-Dr*t)-1)


with open('stelle_parameters.json',"r") as f:
    details = json.load(f)

if details["d_pass"]==0:
    print("No passive particles!")
    exit()
name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"

traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','theta'))
results = msd_pass_func(traj,details)
results.to_csv("../../data/02_processed/msd/"+IO.get_name(details)+"_pass_msd.txt", sep=' ', index=False)

t=results["time"]
msd=results["msd_tot"]
msd_std = results["msd_tot_std"]

nlast=4
#param, pcov = curve_fit(abp, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
param, pcov = curve_fit(lp, t[:-nlast], msd[:-nlast],sigma=msd_std[:-nlast])
perr = np.sqrt(np.diag(pcov))
print(param)
Dv=0.015
gt=9.5
m=1
radius=.5
mass_pass = m * (details["r_pass"]/radius)**3
I=2/5 * mass_pass * details["r_pass"]**2

v0v=details["peclet"]
paramv=[Dv,gt]
print(paramv)

nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

ax[0].errorbar(t,msd,yerr=msd_std,c="b",lw=2,ls="",marker="o", label="all")
ax[0].legend(fontsize=20)
ax[0].plot(t, lp(t, *paramv), linestyle='--',c="b")
ax[0].plot(t, lp(t, *param), linestyle=':',c="r")
#ax[0].plot(t, abp(t, *param), linestyle=':',c="r")
for i in range(nplots):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$msd[\sigma^2]$", fontsize=20)
    ax[i].set_xlabel(r"$t[\tau]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
fig.savefig("../../data/03_analyzed/msd/"+IO.get_name(details)+"_pass_msd.png", bbox_inches="tight",dpi=300)
        