import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import gridspec
import sys
import os
import matplotlib as mpl
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from stelle import IO
import json

def lin(x,x0,m):
    return x0+m*x

def lin1(x,x0):
    return x0-x

def cmapper(max, cm=mpl.cm.plasma,cmap_mod=1.2,min=0):
    norm = mpl.colors.Normalize(vmin=min, vmax=max*cmap_mod)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cmap.set_array([])
    return cmap

def autocorr(x):
    corr = np.correlate(x, x, mode='full')
    result = corr[corr.size//2:]
    return result/result[0]

with open('stelle_parameters.json',"r") as f:
    details = json.load(f)

pe=[1,2,4,8]
rs=np.array([4,5,6,7,8,9,10,12,14,16,18,20])#([5,6,7,8,10,12,14,16,18,20])#([4,5,7,8,10,12,14,16,18,20])
eff_force=np.zeros((len(pe),len(rs)))
eff_force_std=np.zeros((len(pe),len(rs)))
sigma=15
aut_thresh=.01
if details['brownian']==0:
    dt=0.001
else:
    dt=0.00001

for j, a in enumerate(pe):
    details["peclet"] = a
    output = "../../data/02_processed/eff_int/" + IO.get_name(details) + "_eff_int.txt"
    if os.path.isfile(output):
        data = np.loadtxt(output,skiprows=1)
        eff_force[j, :]=data[:,1]
        eff_force_std[j, :]=data[:,2]
        continue
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
        eff_force[j,i]=np.mean(force)
        corr_time=np.argmax(autocorr(force-eff_force[j,i])<aut_thresh)
        eff_force_std[j,i]=np.std(force)/np.sqrt(len(force)/corr_time)
        dd=dr_norm[-1]-dr_norm[0]
        print(dd)
        #plt.plot(autocorr(force-eff_force[i]))
        #plt.show()
    details["r_int"] = 0
    results = pd.DataFrame(columns=["r", "force", "force_std"],
                           data=np.c_[rs, eff_force[j,:], eff_force_std[j,:]])
    results.to_csv(output, sep=' ', index=False)

cmap = cmapper(np.max(pe))
nplots = 1
fig = plt.figure(figsize=(6 * nplots, 6))
gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

ax = []
for i in range(nplots):
    ax.append(fig.add_subplot(gs[:, i:i + 1]))

for j in range(len(pe)):
    ax[0].errorbar(rs,eff_force[j,:],yerr=eff_force_std[j,:],c=cmap.to_rgba(pe[j]),lw=2,ls="",marker="o", label=pe[j])
    param, pcov = curve_fit(lin,rs, np.log(np.abs(eff_force[j,:])), sigma=np.log(eff_force_std[j,:]) / np.abs(eff_force[j,:]))
    perr = np.sqrt(np.diag(pcov))
    ax[0].plot(rs, np.exp(lin(rs, *param)), linestyle='--',c=cmap.to_rgba(pe[j]))

    #param, pcov = curve_fit(lin1,np.log(rs), np.log(np.abs(eff_force[j,:])), sigma=np.log(eff_force_std[j,:]) / np.abs(eff_force[j,:]))
    #perr = np.sqrt(np.diag(pcov))
    #ax[0].plot(rs, np.exp(lin1(np.log(rs), *param)), linestyle='--',c=cmap.to_rgba(pe[j]))

for i in range(nplots):
    #ax[i].set_xscale('log')
    #ax[i].set_yscale('log')
    ax[i].set_ylim(0,)
    ax[i].legend(fontsize=20, frameon=False, title="Pe", title_fontsize=20)
    ax[i].set_ylabel(r"$f[m\sigma/\tau^2]$", fontsize=20)
    ax[i].set_xlabel(r"$r[\sigma]$", fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=20)
plt.show()
details["r_int"] = 0
details["peclet"] = 0
fig.savefig("../../data/03_analyzed/eff_int/"+IO.get_name(details)+"_eff_int.png", bbox_inches="tight",dpi=300)
        