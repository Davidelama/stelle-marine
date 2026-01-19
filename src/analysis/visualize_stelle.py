#!/usr/bin/env python
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec
from scipy.integrate import simps

proj_name = 'stelle'
home = Path(__file__).resolve().home()
root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent  #home / (dir_prj+proj_name)
sys.path.insert(0, str(root_dir / 'src'))
plt.rcParams["text.usetex"]= True
plt.rc('font',**{'family':'serif','serif':['Times']})
cmap_mod=1.2

def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

def lin(x, x0):
    return x0 + x

def power_fit(x, x0, m):
    return x0 + m*x


def linpol(N):
    return 1.43196*N**(.588)*(1-.33039*N**(-.2228))

def dcy(x, x0):
    return x0 - 1.5*x

def fifth_root(x, x0):
    return x0 + .206 * x


def three_fifth_root(x, x0):
    return x0 + .588 * x


def free_chain(x, f, b):
    b * f ** (1 / 2) / x


def self_avoidance(x, f, b):
    return b * f ** .65 * x ** (-1.3)

def self_avoidance_nof(x, b):
    return b * x ** (-1.3)


def multiplication_factor(x, b):
    return b * x

def cmapper(max, cm=mpl.cm.plasma,cmap_mod=1.2):
    norm = mpl.colors.Normalize(vmin=0, vmax=max*cmap_mod)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cmap.set_array([])
    return cmap

col_stat = {}
col_asph = {}
col_armrad = {}
col_rad = {}
col_form = {}
col_bin = {}
rg_f = np.array([])
rg_rc = np.array([])
rg_n = np.array([])
rg_mol = np.array([])
rg_rb = np.array([])
rg_rcb = np.array([])
rg_pe = np.array([])
rg_rf = np.array([])
rg_ri = np.array([])
rg_Dt = np.array([])
rg_Dr = np.array([])
rg_gm = np.array([])
rg_ctc = np.array([])
rg_bro = np.array([])
rg_seed = np.array([])
rg_dts = np.array([])
rgs = pd.DataFrame()
equil = 30

from stelle.analysis_pipeline import AnalysisPipeline

if __name__ == '__main__':
    input_dir = root_dir / 'data/02_processed/stelle'
    output_dir = root_dir / ('data/03_analyzed/stelle')
    output_dir.mkdir(parents=True, exist_ok=True)
    for sim in sorted(input_dir.glob('*')):
        #print(sim.name)
        if sim.name == "old" or sim.name==".DS_Store":
            continue
        outdir = output_dir / sim
        indir = input_dir / sim
        outdir.mkdir(parents=True, exist_ok=True)
        #for sim in indir.glob('dai_*'):
        #print(f"Now visualizing {sim}")
        files_pqt = list(sim.glob('*.pqt'))
        files_binning = list(sim.glob('*binning.csv'))
        pipeline = AnalysisPipeline("mar_" + sim.name)
        #if not (pipeline.graft in graft_want and pipeline.n_beads in n_want and pipeline.functionality in f_want):
        #    continue
        if len(files_pqt) > 0:
            for file_pqt in files_pqt:
                static = pd.read_parquet(file_pqt)
                col_stat[sim.name] = static["gyration"]
                col_asph[sim.name] = static["asphericity"]
                col_armrad[sim.name] = static["arm_gyration"]
                col_rad[sim.name] = static["rad_density"]
                col_form[sim.name] = static["form_factor"]
                rg_f = np.append(rg_f, pipeline.details["functionality"])
                rg_rc = np.append(rg_rc, pipeline.details["r_core"])
                rg_n = np.append(rg_n, pipeline.details["n_beads"])
                rg_mol = np.append(rg_mol, pipeline.details["n_mol"])
                rg_seed = np.append(rg_seed, pipeline.details["seed_start"])
                rg_rb = np.append(rg_rb, pipeline.details["r_bond"])
                rg_rcb = np.append(rg_rcb, pipeline.details["r_cbond"])
                rg_pe = np.append(rg_pe, pipeline.details["peclet"])
                rg_rf = np.append(rg_rf, pipeline.details["r_conf"])
                rg_ri = np.append(rg_ri, pipeline.details["r_int"])
                rg_Dt = np.append(rg_Dt, pipeline.details["Dt"])
                rg_Dr = np.append(rg_Dr, pipeline.details["Dr"])
                rg_gm = np.append(rg_gm, pipeline.details["gamma"])
                rg_ctc = np.append(rg_ctc, pipeline.details["contact"])
                rg_bro = np.append(rg_bro, pipeline.details["brownian"])
                rg_dts = np.append(rg_dts, np.shape(static["gyration"]))
                if np.shape(static["gyration"])[0] < 00:
                    print(sim.name)
                    print(np.shape(static["gyration"]))
                    static["gyration"].iloc[0:].plot()
                    plt.axvline(equil)
                    #plt.xscale("log")
                    plt.show()
        if len(files_binning) > 0:
            for file_binning in files_binning:
                bins = pd.read_csv(file_binning)
                #all_bin[sim.name] = bins["rad_density"]
                col_bin[sim.name] = bins["rad_density"]

        else:
            print(f'---Missing data in {sim}, skipping.')

diam=15

all_static = pd.concat([pd.Series(data, name=key) for key, data in col_stat.items()], axis=1)
all_asphericity = pd.concat([pd.Series(data, name=key) for key, data in col_asph.items()], axis=1)
all_armgyration = pd.concat([pd.Series(data, name=key) for key, data in col_armrad.items()], axis=1)
all_rad = pd.concat([pd.Series(data, name=key) for key, data in col_rad.items()], axis=1) / diam
all_form = pd.concat([pd.Series(data, name=key) for key, data in col_form.items()], axis=1)
all_bin = pd.concat([pd.Series(data, name=key) for key, data in col_bin.items()], axis=1) * diam

rgs["f"] = rg_f.astype(int)
rgs["n"] = rg_n.astype(int)
rgs["rc"] = rg_rc*diam
rgs["seed"] = rg_seed
rgs["r_conf"] = rg_mol*diam
rgs["r_bond"] = rg_rb*diam
rgs["r_cbond"] = rg_rcb*diam
rgs["peclet"] = rg_pe
rgs["r_conf"] = rg_rf*diam
rgs["r_int"] = rg_ri*diam
rgs["Dt"] = rg_Dt*diam*diam
rgs["Dr"] = rg_Dr
rgs["gamma"] = rg_gm
rgs["contact"] = rg_ctc.astype(int)
rgs["brownian"] = rg_bro.astype(int)
rgs["rg_mean"] = all_static.iloc[equil:, :].mean().values*diam
rgs["rg_std"] = all_static.iloc[equil:, :].std().values*diam
rgs["asph_mean"] = all_asphericity.iloc[equil:, :].mean().values
rgs["asph_std"] = all_asphericity.iloc[equil:, :].std().values
rgs["armrg_mean"] = all_armgyration.iloc[equil:, :].mean().values*diam
rgs["armrg_std"] = all_armgyration.iloc[equil:, :].std().values*diam
rgs["rg_mean_off"] = rgs["rg_mean"] - rgs["rc"]
rgs["rg_ratio_mean"] = rgs["rg_mean_off"]/rgs["armrg_mean"]
rgs["rg_ratio_std"] = np.sqrt((rgs["rg_mean_off"]/rgs["armrg_mean"]*rgs["armrg_std"])**2+rgs["rg_mean_off"]**2)/rgs["armrg_mean"]
#rad_stds=pd.DataFrame()

dt = 1e-4


def R_g_n_fig(n_want=5):
    nplots = 1

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for i in range(nplots):

        sel = rgs[(rgs.n == n_want) & (rgs.contact == 1) & (rgs.brownian == 0)]
        ax[i].errorbar(sel.f.values * (i + 1), sel.rg_mean.values, yerr=sel.rg_std.values, c="b", linestyle="",
                           marker="o")


    ax[0].set_ylabel(r"$R_g[mm]$", fontsize=30)  #+.5
    ax[0].set_xlabel(r"$f$", fontsize=30)
    #ax[0].legend(fontsize=25, frameon=False, loc="lower right")
    for i in range(nplots):

        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,)
        ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)

    fig.savefig(output_dir / f"R_g_n{n_want:d}.png", bbox_inches="tight",dpi=300)

def arm_R_g_n_fig(n_want=5):
    nplots = 1

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for i in range(nplots):

        sel = rgs[(rgs.n == n_want) & (rgs.contact == 1) & (rgs.brownian == 0)]
        ax[i].errorbar(sel.f.values * (i + 1), sel.armrg_mean.values, yerr=sel.armrg_std.values, c="b", linestyle="",
                           marker="o")


    ax[0].set_ylabel(r"$R_g[mm]$", fontsize=30)  #+.5
    ax[0].set_xlabel(r"$f$", fontsize=30)
    #ax[0].legend(fontsize=25, frameon=False, loc="lower right")
    for i in range(nplots):

        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,)
        ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)

    fig.savefig(output_dir / f"arm_R_g_n{n_want:d}.png", bbox_inches="tight",dpi=300)

def asph_n_fig(n_want=5):
    nplots = 1

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for i in range(nplots):

        sel = rgs[(rgs.n == n_want) & (rgs.contact == 1) & (rgs.brownian == 0)]
        ax[i].errorbar(sel.f.values * (i + 1), sel.asph_mean.values, yerr=sel.asph_std.values, c="b", linestyle="",
                           marker="o")


    ax[0].set_ylabel(r"$A$", fontsize=30)  #+.5
    ax[0].set_xlabel(r"$f$", fontsize=30)
    #ax[0].legend(fontsize=25, frameon=False, loc="lower right")
    for i in range(nplots):

        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,)
        ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)

    fig.savefig(output_dir / f"asph_n{n_want:d}.png", bbox_inches="tight",dpi=300)

def rad_n_fig(n_want=5):
    f_max=6
    cmap = cmapper(f_max)
    nplots = 1

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("dai_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        rad_means = all_rad[column].iloc[equil:].mean() / (2 * np.pi * all_bin[column])

        if fake_details["n_beads"] == n_want:
            ax[0].plot(all_bin[column], rad_means, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_xlabel(r"$r[mm]$", fontsize=25)

    for i in range(nplots):
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)

        ax[i].set_ylabel(r"$\phi(r)[mm^{-2}]$", fontsize=25)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

    fig.savefig(output_dir / f"rad_n{n_want:d}.png", bbox_inches="tight",dpi=300)

def form_n_fig(n_want=5):
    f_max=6
    cmap = cmapper(f_max)
    nplots = 1

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("dai_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        try:
            form_means_norm = all_form[column].iloc[equil:].mean()
            form_means = all_form[column].iloc[equil:].mean() * (fake_f * fake_details["n_beads"] + 1)
        except:
            continue
        k_vals = np.logspace(-2, 3, num=len(form_means_norm))/diam
        if fake_details["n_beads"] == n_want:
            ax[0].plot(k_vals, form_means_norm, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_xlabel(r"$q[mm^{-1}]$", fontsize=30)

    for i in range(nplots):
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)

        ax[i].set_ylabel(r"$S(q)/(N-1)$", fontsize=30)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

    fig.savefig(output_dir / f"form_n{n_want:d}.png", bbox_inches="tight",dpi=300)



letters = ["a", "b", "c", "d"]

R_g_n_fig()
arm_R_g_n_fig()
asph_n_fig()
rad_n_fig()
form_n_fig()

#plt.show()