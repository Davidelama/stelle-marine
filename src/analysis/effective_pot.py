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

def autocorr(x):
    corr = np.correlate(x, x, mode='full')
    result = corr[corr.size//2:]
    return result/result[0]

with open('stelle_parameters.json',"r") as f:
    details = json.load(f)

rs=np.array([6,7,8,10,12,14,16,18,20])#([5,6,7,8,10,12,14,16,18,20])#([4,5,7,8,10,12,14,16,18,20])
eff_force=np.zeros(len(rs))
eff_force_std=np.zeros(len(rs))
sigma=15
aut_thresh=.01
if details['brownian']==0:
    dt=0.001
else:
    dt=0.0001

for i, r in enumerate(rs):
    details["r_int"] = r
    name = "../../data/01_raw/stelle/" + IO.get_name(details) + "/" + IO.get_name(details) + "_forces.dat"

    data = np.loadtxt(name, skiprows=1)
    t = data[:,0]*dt
    r1 = data[:, 4:6]
    r2 = data[:, 9:11]
    f1 = data[:, 2:4]
    f2 = data[:, 7:9]

    dr=r2-r1
    df=f2-f1

    dr_norm=np.sqrt(np.einsum('ij,ij->i', dr, dr))
    force=np.einsum('ij,ij->i', dr, df)/dr_norm
    eff_force[i]=np.mean(force)
    corr_time=np.argmax(autocorr(force-eff_force[i])<aut_thresh)
    eff_force_std[i]=np.std(force)/np.sqrt(len(force)/corr_time)
    dd=dr_norm[-1]-dr_norm[0]
    print(dd)
    #plt.plot(autocorr(force-eff_force[i]))
    #plt.show()

nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

ax[0].errorbar(rs,eff_force,yerr=eff_force_std,c="b",lw=2,ls="",marker="o", label="all")
ax[0].legend(fontsize=20)
#ax[0].plot(t, ap(t, *paramv), linestyle='--',c="b")
#ax[0].plot(t, ap(t, *param), linestyle=':',c="r")
for i in range(nplots):
    #ax[i].set_xscale('log')
    #ax[i].set_yscale('log')
    ax[i].set_ylabel(r"$f[m\sigma/\tau^2]$", fontsize=20)
    ax[i].set_xlabel(r"$r[\sigma]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
details["r_int"] = 0
fig.savefig("../../data/03_analyzed/eff_int/"+IO.get_name(details)+"_eff_int.png", bbox_inches="tight",dpi=300)
        