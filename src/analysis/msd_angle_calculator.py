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
from stelle.stelle_analysis import rmsd_func


with open('stelle_parameters.json',"r") as f:
    details = json.load(f)


name_unwrap="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_angle_unwrap.lammpstrj"
traj_unwrap = IO.reconstruct_traj([name_unwrap], cols=('at_id','mux','muy'))

name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+"_msd.lammpstrj"
traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','mux','muy'))

results = rmsd_func(traj, traj_unwrap,details)
results.to_csv("../../data/02_processed/msd/"+IO.get_name(details)+"_angle_msd.txt", sep=' ', index=False)

t=results["time"]
msd_c=results["angle_msd_core"]
msd_c_std=results["angle_msd_core_std"]

nlast=4

nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

ax[0].errorbar(t,msd_c,yerr=msd_c_std,c="r",lw=2,ls="",marker="s", label="core")
ax[0].legend(fontsize=20)
ax[0].plot(t,t,c="k")
for i in range(nplots):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$msd[$rad$^2]$", fontsize=20)
    ax[i].set_xlabel(r"$t[\tau]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
fig.savefig("../../data/03_analyzed/msd/"+IO.get_name(details)+"_angle_msd.png", bbox_inches="tight",dpi=300)
        