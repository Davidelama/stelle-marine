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

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return np.abs(a-b) <= np.max(rel_tol * np.max(np.abs(a), np.abs(b)), abs_tol)

def autocorr(x):
    corr = np.correlate(x, x, mode='full')
    result = corr[corr.size//2:]
    return result/result[0]

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

def sqr_root(x, x0):
    return x0 + .5 * x

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

def cmapper(max, cm=mpl.cm.plasma,cmap_mod=1.2,min=0):
    norm = mpl.colors.Normalize(vmin=min, vmax=max*cmap_mod)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cmap.set_array([])
    return cmap

col_stat = {}
col_asph = {}
col_armete = {}
col_vctc = {}
col_clrt = {}
col_chull = {}
col_rad = {}
col_form = {}
col_bin = {}
col_time = {}
col_msd = {}
col_msd_std = {}
col_msdc = {}
col_msdc_std = {}
col_rmsd = {}
col_rmsd_std = {}
col_cdists = {}
col_cforces = {}
col_hcr = {}
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
rg_rll = np.array([])
rg_bro = np.array([])
rg_dpass = np.array([])
rg_rpass = np.array([])
rg_Dtpass = np.array([])
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
        files_pqt = list(sim.glob('*static_properties.pqt'))
        files_cdf = list(sim.glob('*cdf_properties.pqt'))
        files_binning = list(sim.glob('*binning.csv'))
        files_dynamical = list(sim.glob('*dynamical_properties.csv'))
        files_interaction = list(sim.glob('*interaction_properties.csv'))
        files_capture = list(sim.glob('*capture_properties.csv'))
        pipeline = AnalysisPipeline("mar_" + sim.name)
        if pipeline.details["r_int"]>0 and pipeline.details["d_pass"]==0:
            continue
        #if not (pipeline.graft in graft_want and pipeline.n_beads in n_want and pipeline.functionality in f_want):
        #    continue
        if len(files_pqt) > 0:
            for file_pqt in files_pqt:
                static = pd.read_parquet(file_pqt)
                col_stat[sim.name] = static["gyration"]
                col_asph[sim.name] = static["asphericity"]
                col_armete[sim.name] = static["arm_ete"]
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
                rg_rll = np.append(rg_rll, pipeline.details["rolling"])
                rg_bro = np.append(rg_bro, pipeline.details["brownian"])
                rg_dpass = np.append(rg_dpass, pipeline.details["d_pass"])
                rg_rpass = np.append(rg_rpass, pipeline.details["r_pass"])
                rg_Dtpass = np.append(rg_Dtpass, pipeline.details["Dt_pass"])
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

        if len(files_dynamical) > 0:
            for file_dynamical in files_dynamical:
                dyn = pd.read_csv(file_dynamical)
                col_time[sim.name] = dyn["time"]
                col_msd[sim.name] = dyn["msd_tot"]
                col_msd_std[sim.name] = dyn["msd_tot_std"]
                col_msdc[sim.name] = dyn["msd_core"]
                col_msdc_std[sim.name] = dyn["msd_core_std"]
                col_rmsd[sim.name] = dyn["angle_msd_core"]
                col_rmsd_std[sim.name] = dyn["angle_msd_core_std"]

        if len(files_interaction) > 0:
            for file_interaction in files_interaction:
                itr = pd.read_csv(file_interaction)
                col_vctc[sim.name] = itr["contact_ratio"]
                col_clrt[sim.name] = itr["close_ratio"]
                col_chull[sim.name] = itr["hull_intersection"]

        """if len(files_cdf) > 0:
            for file_cdf in files_cdf:
                cdf = pd.read_parquet(file_cdf)
                col_cdists[sim.name] = cdf["dist"]
                col_cforces[sim.name] = cdf["force"]"""

        if len(files_capture) > 0:
            for file_capture in files_capture:
                cap = pd.read_csv(file_capture)
                col_hcr[sim.name] = cap["hull_cap_rat"]


        else:
            print(f'---Missing data in {sim}, skipping.')

diam=15
dt=1e-4
tf=dt*1e6
ttot=dt*2e9
tdec=10*60
ndecorr=ttot/tdec

all_static = pd.concat([pd.Series(data, name=key) for key, data in col_stat.items()], axis=1)
all_asphericity = pd.concat([pd.Series(data, name=key) for key, data in col_asph.items()], axis=1)
all_armete = pd.concat([pd.Series(data, name=key) for key, data in col_armete.items()], axis=1)
all_vctc = pd.concat([pd.Series(data, name=key) for key, data in col_vctc.items()], axis=1)
all_clrt = pd.concat([pd.Series(data, name=key) for key, data in col_clrt.items()], axis=1)
all_chull = pd.concat([pd.Series(data, name=key) for key, data in col_chull.items()], axis=1)
all_rad = pd.concat([pd.Series(data, name=key) for key, data in col_rad.items()], axis=1)
all_form = pd.concat([pd.Series(data, name=key) for key, data in col_form.items()], axis=1)
all_bin = pd.concat([pd.Series(data, name=key) for key, data in col_bin.items()], axis=1) * diam
all_time = pd.concat([pd.Series(data, name=key) for key, data in col_time.items()], axis=1)
all_msd = pd.concat([pd.Series(data, name=key) for key, data in col_msd.items()], axis=1) * diam**2
all_msd_std = pd.concat([pd.Series(data, name=key) for key, data in col_msd_std.items()], axis=1) * diam**2
all_msdc = pd.concat([pd.Series(data, name=key) for key, data in col_msdc.items()], axis=1) * diam**2
all_msdc_std = pd.concat([pd.Series(data, name=key) for key, data in col_msdc_std.items()], axis=1) * diam**2
all_rmsd = pd.concat([pd.Series(data, name=key) for key, data in col_rmsd.items()], axis=1)
all_rmsd_std = pd.concat([pd.Series(data, name=key) for key, data in col_rmsd_std.items()], axis=1)
#all_cdists = pd.concat([pd.Series(data, name=key) for key, data in col_cdists.items()], axis=1) * diam
#all_cforces = pd.concat([pd.Series(data, name=key) for key, data in col_cforces.items()], axis=1) * diam
all_hcr = pd.concat([pd.Series(data, name=key) for key, data in col_hcr.items()], axis=1)

rgs["f"] = rg_f.astype(int)
rgs["n"] = rg_n.astype(int)
rgs["rc"] = rg_rc*diam
rgs["mols"] = rg_mol.astype(int)
rgs["seed"] = rg_seed
rgs["r_bond"] = rg_rb*diam
rgs["r_cbond"] = rg_rcb*diam
rgs["peclet"] = rg_pe
rgs["r_conf"] = rg_rf*diam
rgs["r_int"] = rg_ri*diam
rgs["Dt"] = rg_Dt*diam*diam
rgs["Dr"] = rg_Dr
rgs["gamma"] = rg_gm
rgs["contact"] = rg_ctc.astype(int)
rgs["roll"] = rg_rll.astype(int)
rgs["brownian"] = rg_bro.astype(int)
rgs["d_pass"] = rg_dpass
rgs["r_pass"] = rg_rpass
rgs["Dt_pass"] = rg_Dtpass
rgs["rg_mean"] = all_static.iloc[equil:, :].mean().values*diam
rgs["rg_std"] = all_static.iloc[equil:, :].std().values*diam/np.sqrt(ndecorr)
rgs["v_ctc_mean"] = all_vctc.iloc[equil:, :].mean().values
rgs["v_ctc_std"] = all_vctc.iloc[equil:, :].std().values/np.sqrt(ndecorr)
rgs["clrt_mean"] = all_clrt.iloc[equil:, :].mean().values
rgs["clrt_std"] = all_clrt.iloc[equil:, :].std().values/np.sqrt(ndecorr)
rgs["chull_mean"] = all_chull.iloc[equil:, :].mean().values
rgs["chull_std"] = all_chull.iloc[equil:, :].std().values/np.sqrt(ndecorr)
rgs["asph_mean"] = all_asphericity.iloc[equil:, :].mean().values
rgs["asph_std"] = all_asphericity.iloc[equil:, :].std().values/np.sqrt(ndecorr)
rgs["armete_mean"] = all_armete.iloc[equil:, :].mean().values*diam
rgs["armete_std"] = all_armete.iloc[equil:, :].std().values*diam/np.sqrt(ndecorr)
rgs["rg_mean_off"] = rgs["rg_mean"] - rgs["rc"]
#rad_stds=pd.DataFrame()

dt = 1e-4


def R_g_n_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 3
    fmin=8
    arr = np.array([fmin, 16])
    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap = cmapper(np.max(rgs.f), min=np.min(rgs.f))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for fval in np.unique(rgs.f):
            sel = rgs[(rgs.peclet == pe) & (rgs.f == fval) &  (rgs.contact == ctc) & (rgs.roll == rll)  & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.n.values, sel.rg_mean.values, yerr=sel.rg_std.values, c=cmap.to_rgba(fval), linestyle="",
                           marker="o",label=f"{fval}")

            fcond=sel.n.values>fmin
            param, pcov = curve_fit(sqr_root, np.log(sel.n.values)[fcond], np.log(sel.rg_mean.values)[fcond],
                                    sigma=np.log(sel.rg_std.values)[fcond] / sel.rg_mean.values[fcond])
            perr = np.sqrt(np.diag(pcov))
            ax[i].plot(arr, np.exp(sqr_root(np.log(arr), *param)), linestyle='--', c=cmap.to_rgba(fval))


    ax[0].set_ylabel(r"$R_g[mm]$", fontsize=30)  #+.5

    ax[0].legend(title=r"$f$", title_fontsize=25, fontsize=25)

    ax[-1].text(.55, .75, r"$\propto N^{1/2}$", fontsize=25, transform=ax[-1].transAxes)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(30,100)
        ax[i].plot(arr, arr ** (1/2) * 20, c="k", ls="--", lw=3)

        #ax[i].set_xlim(0,)
        ax[i].set_xlabel(r"$n$", fontsize=30)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"R_g_n_pe{pe}.png", bbox_inches="tight",dpi=300)

def R_g_f_fig(ri=0,rf=10,nmol=1, pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.rg_mean.values, yerr=sel.rg_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$R_g[mm]$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,100)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"R_g_f_pe{pe}.png", bbox_inches="tight",dpi=300)

def R_g_nf_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 1
    fmin = 15
    arr = np.linspace(fmin, 90)

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for i in range(nplots):
        for j in [2]:#range(3):
            ctc = 0
            rll = 0
            if j > 0:
                ctc = 1
            if j > 1:
                rll = 1
            sel = rgs[(rgs.peclet == pe) & (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol)] & (rgs.d_pass == dpass)
            ax[i].errorbar(sel.f.values*sel.n.values, sel.rg_mean.values, yerr=sel.rg_std.values, c=colors[j], linestyle="",
                           marker=ms[j],label=labs[j])

            fcond=sel.n.values*sel.f.values>fmin
            param, pcov = curve_fit(sqr_root, np.log(sel.n.values*sel.f.values)[fcond], np.log(sel.rg_mean.values)[fcond],
                                    sigma=np.log(sel.rg_std.values)[fcond] / sel.rg_mean.values[fcond])
            perr = np.sqrt(np.diag(pcov))
            ax[i].plot(arr, np.exp(sqr_root(np.log(arr), *param)), linestyle='--', c=colors[j])


    ax[0].set_ylabel(r"$R_g[mm]$", fontsize=30)  #+.5

    ax[0].legend(fontsize=25, frameon=False)#, loc="lower right")
    for i in range(nplots):
        ax[i].set_xlabel(r"$f\times n$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(30,90)
        #ax[i].set_xlim(0,)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"R_g_nf_pe{pe}.png", bbox_inches="tight",dpi=300)

def arm_ete_n_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap = cmapper(np.max(rgs.f), min=np.min(rgs.f))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for fval in np.unique(rgs.f):
            sel = rgs[(rgs.peclet == pe) & (rgs.f == fval) &  (rgs.contact == ctc) & (rgs.roll == rll)  & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.n.values, sel.armete_mean.values, yerr=sel.armete_std.values, c=cmap.to_rgba(fval), linestyle="",
                           marker="o",label=f"{fval}")


    ax[0].set_ylabel(r"$R_{ee}[mm]$", fontsize=30)  #+.5

    ax[0].legend(title=r"$f$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_title(labs[i], fontsize=25)
        ax[i].set_ylim(0,60)
        #ax[i].set_xlim(0,)
        ax[i].set_xlabel(r"$n$", fontsize=30)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"arm_ete_n_pe{pe}.png", bbox_inches="tight",dpi=300)

def arm_ete_f_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.armete_mean.values, yerr=sel.armete_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$R_{ee}[mm]$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,60)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"arm_ete_f_pe{pe}.png", bbox_inches="tight",dpi=300)

def arm_ete_nf_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 1

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for i in range(nplots):
        for j in range(3):
            ctc = 0
            rll = 0
            if j > 0:
                ctc = 1
            if j > 1:
                rll = 1
            sel = rgs[(rgs.peclet == pe) & (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values*sel.n.values, sel.armete_mean.values, yerr=sel.armete_std.values, c=colors[j], linestyle="",
                           marker=ms[j],label=labs[j])


    ax[0].set_ylabel(r"$R_{ee}[mm]$", fontsize=30)  #+.5

    ax[0].legend(fontsize=25, frameon=False)#, loc="lower right")
    for i in range(nplots):
        ax[i].set_xlabel(r"$f\times n$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,60)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"arm_ete_nf_pe{pe}.png", bbox_inches="tight",dpi=300)

def asph_f_fig(ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.asph_mean.values, yerr=sel.asph_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$A$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,.2)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"asph_f_pe{pe}.png", bbox_inches="tight",dpi=300)

def vctc_n_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3
    fmin=8
    arr = np.array([fmin, 16])
    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap = cmapper(np.max(rgs.f), min=np.min(rgs.f))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for fval in np.unique(rgs.f):
            sel = rgs[(rgs.peclet == pe) & (rgs.f == fval) &  (rgs.contact == ctc) & (rgs.roll == rll)  & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.n.values, sel.v_ctc_mean.values, yerr=sel.v_ctc_std.values, c=cmap.to_rgba(fval), linestyle="",
                           marker="o",label=f"{fval}")

            """fcond=sel.n.values>fmin
            param, pcov = curve_fit(sqr_root, np.log(sel.n.values)[fcond], np.log(sel.rg_mean.values)[fcond],
                                    sigma=np.log(sel.rg_std.values)[fcond] / sel.rg_mean.values[fcond])
            perr = np.sqrt(np.diag(pcov))
            ax[i].plot(arr, np.exp(sqr_root(np.log(arr), *param)), linestyle='--', c=cmap.to_rgba(fval))
"""

    ax[0].set_ylabel(r"$c_v$", fontsize=30)  #+.5

    ax[0].legend(title=r"$f$", title_fontsize=25, fontsize=25)

    #ax[-1].text(.55, .75, r"$\propto N^{1/2}$", fontsize=25, transform=ax[-1].transAxes)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.3)
        #ax[i].plot(arr, arr ** (1/2) * 20, c="k", ls="--", lw=3)

        #ax[i].set_xlim(0,)
        ax[i].set_xlabel(r"$n$", fontsize=30)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"c_v_n_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def vctc_f_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.v_ctc_mean.values, yerr=sel.v_ctc_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$c_v$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.3)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"c_v_f_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def clrt_n_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3
    fmin=8
    arr = np.array([fmin, 16])
    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap = cmapper(np.max(rgs.f), min=np.min(rgs.f))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for fval in np.unique(rgs.f):
            sel = rgs[(rgs.peclet == pe) & (rgs.f == fval) &  (rgs.contact == ctc) & (rgs.roll == rll)  & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.n.values, sel.clrt_mean.values, yerr=sel.clrt_std.values, c=cmap.to_rgba(fval), linestyle="",
                           marker="o",label=f"{fval}")

            """fcond=sel.n.values>fmin
            param, pcov = curve_fit(sqr_root, np.log(sel.n.values)[fcond], np.log(sel.rg_mean.values)[fcond],
                                    sigma=np.log(sel.rg_std.values)[fcond] / sel.rg_mean.values[fcond])
            perr = np.sqrt(np.diag(pcov))
            ax[i].plot(arr, np.exp(sqr_root(np.log(arr), *param)), linestyle='--', c=cmap.to_rgba(fval))
"""

    ax[0].set_ylabel(r"$r_c$", fontsize=30)  #+.5

    ax[0].legend(title=r"$f$", title_fontsize=25, fontsize=25)

    #ax[-1].text(.55, .75, r"$\propto N^{1/2}$", fontsize=25, transform=ax[-1].transAxes)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.015)
        #ax[i].plot(arr, arr ** (1/2) * 20, c="k", ls="--", lw=3)

        #ax[i].set_xlim(0,)
        ax[i].set_xlabel(r"$n$", fontsize=30)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"r_c_n_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def clrt_f_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.clrt_mean.values, yerr=sel.clrt_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$r_c$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.015)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"r_c_f_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def chull_n_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3
    fmin=8
    arr = np.array([fmin, 16])
    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap = cmapper(np.max(rgs.f), min=np.min(rgs.f))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for fval in np.unique(rgs.f):
            sel = rgs[(rgs.peclet == pe) & (rgs.f == fval) &  (rgs.contact == ctc) & (rgs.roll == rll)  & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.n.values, sel.chull_mean.values, yerr=sel.chull_std.values, c=cmap.to_rgba(fval), linestyle="",
                           marker="o",label=f"{fval}")

            """fcond=sel.n.values>fmin
            param, pcov = curve_fit(sqr_root, np.log(sel.n.values)[fcond], np.log(sel.rg_mean.values)[fcond],
                                    sigma=np.log(sel.rg_std.values)[fcond] / sel.rg_mean.values[fcond])
            perr = np.sqrt(np.diag(pcov))
            ax[i].plot(arr, np.exp(sqr_root(np.log(arr), *param)), linestyle='--', c=cmap.to_rgba(fval))
"""

    ax[0].set_ylabel(r"$r_c$", fontsize=30)  #+.5

    ax[0].legend(title=r"$f$", title_fontsize=25, fontsize=25)

    #ax[-1].text(.55, .75, r"$\propto N^{1/2}$", fontsize=25, transform=ax[-1].transAxes)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.12)
        #ax[i].plot(arr, arr ** (1/2) * 20, c="k", ls="--", lw=3)

        #ax[i].set_xlim(0,)
        ax[i].set_xlabel(r"$n$", fontsize=30)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"c_h_n_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def chull_f_fig(ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    nplots = 3

    fig = plt.figure(figsize=(5*nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    cmap=cmapper(np.max(rgs.n),min=np.min(rgs.n))
    for i in range(nplots):
        ctc=0
        rll=0
        if i>0:
            ctc=1
        if i>1:
            rll=1
        for nval in np.unique(rgs.n):
            sel = rgs[(rgs.peclet == pe) & (rgs.n == nval) &  (rgs.contact == ctc) & (rgs.roll == rll) & (rgs.brownian == 0) & (rgs.r_conf == rf*diam) & (rgs.r_int == ri*diam) & (rgs.mols == nmol) & (rgs.d_pass == dpass)]
            ax[i].errorbar(sel.f.values, sel.chull_mean.values, yerr=sel.chull_std.values, c=cmap.to_rgba(nval), linestyle="",
                           marker="o",label=f"{nval}")


    ax[0].set_ylabel(r"$r_c$", fontsize=30)  #+.5

    ax[0].legend(title=r"$n$", title_fontsize=25, fontsize=25)
    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].set_xlabel(r"$f$", fontsize=30)
        ax[i].text(.05,.9,letters[i],fontsize=30, transform=ax[i].transAxes)
        ax[i].set_ylim(0,0.12)
        #ax[i].set_xlim(0,)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        ax[i].tick_params(axis='y', which='major', labelsize=30)
        ax[i].tick_params(axis='x', which='major', labelsize=30)
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"c_h_f_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def rad_n_fig(n_want=5,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        dr=all_bin[column][1]-all_bin[column][0]
        rad_means = all_rad[column].iloc[equil:].mean() / (2 * np.pi * all_bin[column] * dr)*diam**2*np.pi/4

        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].plot(all_bin[column], rad_means, label=fake_f, c=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(all_bin[column], rad_means, label=fake_f, c=cmap.to_rgba(fake_f))
            else:
                ax[2].plot(all_bin[column], rad_means, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$\phi(r)A_p$", fontsize=25)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$r[mm]$", fontsize=25)
        ax[i].set_ylim(0,3.0)
        ax[i].set_xlim(0, 80)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"rad_n{n_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def rad_f_fig(f_want=6,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_n = fake_details["n_beads"]
        dr = all_bin[column][1] - all_bin[column][0]
        rad_means = all_rad[column].iloc[equil:].mean() / (2 * np.pi * all_bin[column] *dr)*diam**2*np.pi/4

        if fake_details["functionality"] == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].plot(all_bin[column], rad_means, label=fake_n, c=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(all_bin[column], rad_means, label=fake_n, c=cmap.to_rgba(fake_n))
            else:
                ax[2].plot(all_bin[column], rad_means, label=fake_n, c=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$n$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$\phi(r)A_p$", fontsize=25)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$r[mm]$", fontsize=25)
        ax[i].set_ylim(0,3.0)
        ax[i].set_xlim(0, 80)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"rad_f{f_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def form_n_fig(n_want=5,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        try:
            form_means_norm = all_form[column].iloc[equil:].mean()
            form_means = all_form[column].iloc[equil:].mean() * (fake_f * fake_details["n_beads"] + 1)
            len(form_means_norm)
        except:
            continue
        k_vals = np.logspace(-2, 3, num=len(form_means_norm))/diam
        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].plot(k_vals, form_means_norm, label=fake_f, c=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(k_vals, form_means_norm, label=fake_f, c=cmap.to_rgba(fake_f))
            else:
                ax[2].plot(k_vals, form_means_norm, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)

    ax[0].set_ylabel(r"$S(q)/(N-1)$", fontsize=30)

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$q[mm^{-1}]$", fontsize=30)
        ax[i].set_ylim(5e-3, 1.1)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"form_n{n_want:d}.png", bbox_inches="tight",dpi=300)

def form_f_fig(f_want=6,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        fake_n = fake_details["n_beads"]
        try:
            form_means_norm = all_form[column].iloc[equil:].mean()
            form_means = all_form[column].iloc[equil:].mean() * (fake_details["functionality"] * fake_n + 1)
            len(form_means_norm)
        except:
            continue
        k_vals = np.logspace(-2, 3, num=len(form_means_norm))/diam
        if fake_f == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].plot(k_vals, form_means_norm, label=fake_n, c=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(k_vals, form_means_norm, label=fake_n, c=cmap.to_rgba(fake_n))
            else:
                ax[2].plot(k_vals, form_means_norm, label=fake_n, c=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)

    ax[0].set_ylabel(r"$S(q)/(N-1)$", fontsize=30)

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$q[mm^{-1}]$", fontsize=30)
        ax[i].set_ylim(2e-3, 1.1)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"form_f{f_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def msd_n_fig(n_want=5,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        time = all_time[column]
        msd = all_msd[column]
        msd_std = all_msd_std[column]

        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$MSD[mm^2]$", fontsize=25)
    ytop=1e8
    if rf>0:
        ytop = 1e5

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-6,ytop)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"msd_n{n_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def msd_f_fig(f_want=6,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        fake_n = fake_details["n_beads"]
        time = all_time[column]
        msd = all_msd[column]
        msd_std = all_msd_std[column]

        if fake_f == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$n$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$MSD[mm^2]$", fontsize=25)
    ytop=1e8
    if rf>0:
        ytop = 1e5

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-6,ytop)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"msd_f{f_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def msdc_n_fig(n_want=5,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        time = all_time[column]
        msd = all_msdc[column]
        msd_std = all_msdc_std[column]

        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$MSD[mm^2]$", fontsize=25)
    ytop=1e8
    if rf>0:
        ytop = 1e5

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-6,ytop)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"msdc_n{n_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def msdc_f_fig(f_want=6,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        fake_n = fake_details["n_beads"]
        time = all_time[column]
        msd = all_msdc[column]
        msd_std = all_msdc_std[column]

        if fake_f == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$n$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$MSD[mm^2]$", fontsize=25)
    ytop=1e8
    if rf>0:
        ytop = 1e5

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-6,ytop)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"msdc_f{f_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def rmsd_n_fig(n_want=5,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        time = all_time[column]
        msd = all_rmsd[column]
        msd_std = all_rmsd_std[column]

        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_f, c=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$RMSD[rad^2]$", fontsize=25)

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-8,1e5)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"rmsd_n{n_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def rmsd_f_fig(f_want=6,ri=0,rf=10,nmol=1,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]
        fake_n = fake_details["n_beads"]
        time = all_time[column]
        msd = all_rmsd[column]
        msd_std = all_rmsd_std[column]

        if fake_f == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))
            else:
                ax[2].errorbar(time, msd,yerr=msd_std, label=fake_n, c=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$n$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$RMSD[rad^2]$", fontsize=25)

    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$t[s]$", fontsize=25)
        ax[i].set_ylim(1e-8,1e8)
        #ax[i].set_xlim(0, 80)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"rmsd_f{f_want:d}_pe{pe}.png", bbox_inches="tight",dpi=300)

def cdists_n_fig(n_want=5,ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.f),min=np.min(rgs.f))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_f = fake_details["functionality"]

        counts, bins = np.histogram(all_cdists[column], bins=40, range=(10, 250))
        bm = (bins[1:] + bins[:-1]) / 2
        bd = (bins[1:] - bins[:-1])
        gr = counts /(2 * np.pi * bm * bd * len(all_cdists[column]))*diam**2*np.pi/4

        if fake_details["n_beads"] == n_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:
                ax[0].plot(bm,gr, label=fake_f, color=cmap.to_rgba(fake_f))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(bm,gr, label=fake_f, color=cmap.to_rgba(fake_f))
            else:
                ax[2].plot(bm,gr, label=fake_f, color=cmap.to_rgba(fake_f))

    ax[0].legend(title=r"$f$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$g(r)A_p$", fontsize=25)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$r[mm]$", fontsize=25)
        ax[i].set_ylim(0,1.2e-3)
        ax[i].set_xlim(0, 250)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"cdists_n{n_want:d}_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def cdists_f_fig(f_want=6,ri=0,rf=10,nmol=2,pe=2.3,dpass=0):
    cmap = cmapper(np.max(rgs.n),min=np.min(rgs.n))
    nplots = 3

    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))

    for column in all_rad:
        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        fake_n = fake_details["n_beads"]
        counts, bins = np.histogram(all_cdists[column], bins=40, range=(10, 250))
        bm = (bins[1:] + bins[:-1]) / 2
        bd = (bins[1:] - bins[:-1])
        gr = counts /(2 * np.pi * bm * bd * len(all_cdists[column]))*diam**2*np.pi/4

        if fake_details["functionality"] == f_want and fake_details["r_conf"] == rf and fake_details["r_int"] == ri and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == dpass:
            if fake_details["contact"] == 0:

                ax[0].plot(bm,gr, label=fake_n, color=cmap.to_rgba(fake_n))
            elif fake_details["contact"] == 1 and fake_details["rolling"] == 0:
                ax[1].plot(bm,gr, label=fake_n, color=cmap.to_rgba(fake_n))
            else:
                ax[2].plot(bm,gr, label=fake_n, color=cmap.to_rgba(fake_n))

    ax[0].legend(title=r"$n$", title_fontsize=20, fontsize=20)
    ax[0].set_ylabel(r"$g(r)A_p$", fontsize=25)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$r[mm]$", fontsize=25)
        ax[i].set_ylim(0,1.2e-3)
        ax[i].set_xlim(0, 250)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"cdists_f{f_want:d}_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def effpot_calc(nmol=2,pe=2.3,n_want=15,aut_thresh=.01):


    fs = np.sort(rgs.f.unique())
    ris = np.sort(rgs.r_int.unique()).astype(int)



    eff_force = np.ones((len(fs),len(ris),3))*np.nan
    eff_force_std = np.ones((len(fs),len(ris), 3))*np.nan
    for column in all_cforces:

        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        if fake_details["r_conf"] == 0 and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] == 0 and fake_details["n_beads"] == n_want:
            m = np.argwhere(fs == fake_details["functionality"])[0][0]
            k=np.argwhere(ris==int(fake_details["r_int"]*diam))[0][0]

            if fake_details["contact"]==0 and fake_details["rolling"]==0:
                i=0
            elif fake_details["contact"]==1 and fake_details["rolling"]==0:
                i=1
            else:
                i=2
            eff_force[m,k,i]=np.mean(all_cforces[column])
            #corr_time=np.argmax(autocorr(all_cforces[column]-eff_force[j,m,k,i])<aut_thresh)
            #eff_force_std[j,m,k,i]=np.std(all_cdists[column])/np.sqrt(len(all_cforces[column])/corr_time)

    np.savetxt(output_dir / f"effpots//effpot_ri_n{n_want}_pe{pe}_m{nmol}.txt", ris, fmt='%d')
    np.savetxt(output_dir / f"effpots//effpot_f_n{n_want}_pe{pe}_m{nmol}.txt", fs, fmt='%d')
    np.save(output_dir / f"effpots//effpot_vals_n{n_want}_pe{pe}_m{nmol}.npy",eff_force)
    np.save(output_dir / f"effpots//effpot_std_n{n_want}_pe{pe}_m{nmol}.npy", eff_force_std)


def effpot_fig(n_want=15, nmol=2, pe=2.3):

    fs=np.loadtxt(output_dir / f"effpots//effpot_f_n{n_want}_pe{pe}_m{nmol}.txt")
    ris=np.loadtxt(output_dir / f"effpots//effpot_ri_n{n_want}_pe{pe}_m{nmol}.txt")
    forces=np.load(output_dir / f"effpots//effpot_vals_n{n_want}_pe{pe}_m{nmol}.npy")
    forces_std = np.load(output_dir / f"effpots//effpot_std_n{n_want}_pe{pe}_m{nmol}.npy")

    nplots = 3
    cmap = cmapper(len(fs))
    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))


    ax[0].set_ylabel(r"$F$", fontsize=25)

    for i in range(nplots):
        for j in range(len(fs)):
            ax[i].errorbar(ris,forces[j,:,i],yerr=forces_std[j,:,i], color=cmap.to_rgba(j),label=fs[j])

    ax[0].legend(title=r"f", title_fontsize=20, fontsize=20)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$r[mm]$", fontsize=25)
        #ax[i].set_ylim(0,1.2e-3)
        #ax[i].set_xlim(0, 250)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"effpot_n{n_want:d}_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def hcr_n_fig(nmol=1,rf=10,pe=2.3,n_want=11):

    ns = np.sort(rgs.n.unique())
    fs = np.sort(rgs.f.unique())
    dpass = np.sort(rgs.d_pass.unique())



    hcr = np.ones((len(ns),len(fs),len(dpass), 3))*np.nan
    hcr_std = np.ones((len(ns),len(fs),len(dpass), 3))*np.nan
    for column in all_hcr:

        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        if fake_details["r_conf"] == rf and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] > 0:
            m = np.argwhere(fs == fake_details["functionality"])[0][0]
            k=np.argwhere(ns==fake_details["n_beads"])[0][0]
            j = np.argwhere(np.abs(dpass- fake_details["d_pass"])<1e-8)[0][0]

            if fake_details["contact"]==0 and fake_details["rolling"]==0:
                i=0
            elif fake_details["contact"]==1 and fake_details["rolling"]==0:
                i=1
            else:
                i=2
            hcr[k,m,j,i]=np.mean(all_hcr[column])
            hcr_std[k,m,j,i]=np.std(all_hcr[column])/np.sqrt(ndecorr)
            #corr_time=np.argmax(autocorr(all_cforces[column]-eff_force[j,m,k,i])<aut_thresh)
            #eff_force_std[j,m,k,i]=np.std(all_cdists[column])/np.sqrt(len(all_cforces[column])/corr_time)

    nplots = 3
    cmap = cmapper(len(fs))
    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))


    ax[0].set_ylabel(r"hcr", fontsize=25)

    for i in range(nplots):
        for j in range(len(fs)):
            if np.isnan(hcr[ns==n_want,j,:,i][0]).all():
                continue
            ax[i].errorbar(dpass,hcr[ns==n_want,j,:,i][0],yerr=hcr_std[ns==n_want,j,:,i][0], color=cmap.to_rgba(j),label=fs[j], linestyle="",
                           marker="o")

    ax[0].legend(title=r"f", title_fontsize=20, fontsize=20)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$\phi$", fontsize=25)
        ax[i].set_ylim(0,0.125)
        #ax[i].set_xlim(0, 250)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"hcr_n{n_want}_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)

def hcr_f_fig(nmol=1,rf=10,pe=2.3,f_want=6):

    ns = np.sort(rgs.n.unique())
    fs = np.sort(rgs.f.unique())
    dpass = np.sort(rgs.d_pass.unique())



    hcr = np.ones((len(ns),len(fs),len(dpass), 3))*np.nan
    hcr_std = np.ones((len(ns),len(fs),len(dpass), 3))*np.nan
    for column in all_hcr:

        fake_pipeline = AnalysisPipeline("mar_" + column)
        fake_details = fake_pipeline.details
        if fake_details["r_conf"] == rf and fake_details["n_mol"] == nmol and fake_details["peclet"] == pe and fake_details["d_pass"] > 0:
            m = np.argwhere(fs == fake_details["functionality"])[0][0]
            k=np.argwhere(ns==fake_details["n_beads"])[0][0]
            j = np.argwhere(np.abs(dpass- fake_details["d_pass"])<1e-8)[0][0]

            if fake_details["contact"]==0 and fake_details["rolling"]==0:
                i=0
            elif fake_details["contact"]==1 and fake_details["rolling"]==0:
                i=1
            else:
                i=2
            hcr[k,m,j,i]=np.mean(all_hcr[column])
            hcr_std[k,m,j,i]=np.std(all_hcr[column])/np.sqrt(ndecorr)
            #corr_time=np.argmax(autocorr(all_cforces[column]-eff_force[j,m,k,i])<aut_thresh)
            #eff_force_std[j,m,k,i]=np.std(all_cdists[column])/np.sqrt(len(all_cforces[column])/corr_time)

    nplots = 3
    cmap = cmapper(len(fs))
    fig = plt.figure(figsize=(5 * nplots, 5))
    gs = gridspec.GridSpec(1, nplots, hspace=0, wspace=0)

    ax = []
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[:, i:i + 1]))


    ax[0].set_ylabel(r"hcr", fontsize=25)

    for i in range(nplots):
        for j in range(len(ns)):
            if np.isnan(hcr[j,fs==f_want,:,i][0]).all():
                continue
            ax[i].errorbar(dpass,hcr[j,fs==f_want,:,i][0],yerr=hcr_std[j,fs==f_want,:,i][0], color=cmap.to_rgba(j),label=ns[j], linestyle="",
                           marker="o")

    ax[0].legend(title=r"n", title_fontsize=20, fontsize=20)


    for i in range(nplots):
        ax[i].set_title(labs[i],fontsize=25)
        ax[i].tick_params(axis='y', which='major', labelsize=25)
        ax[i].tick_params(axis='x', which='major', labelsize=25)
        ax[i].set_xlabel(r"$\phi$", fontsize=25)
        ax[i].set_ylim(0,0.2)
        #ax[i].set_xlim(0, 250)
        #ax[i].set_xscale("log")
        #ax[i].set_yscale("log")
        if i>0:
            ax[i].tick_params(which='both', left=False, labelleft=False)

    fig.savefig(output_dir / f"hcr_f{f_want}_pe{pe}_m{nmol}.png", bbox_inches="tight",dpi=300)


letters = ["a", "b", "c", "d"]
ms = ["s","o","^","*"]
colors = ["r", "g", "b", "k"]
labs = ["no friction", "contact", "rolling"]
molwant=1
rfwant=10

#effpot_calc()
#effpot_fig()
hcr_f_fig()
hcr_f_fig(pe=0)
hcr_f_fig(f_want=3)
hcr_f_fig(f_want=3,pe=0)
plt.show()

"""R_g_n_fig(rf=rfwant, nmol=molwant)
R_g_f_fig(rf=rfwant, nmol=molwant)
R_g_nf_fig(rf=rfwant, nmol=molwant)
arm_ete_n_fig(rf=rfwant, nmol=molwant)
arm_ete_f_fig(rf=rfwant, nmol=molwant)
arm_ete_nf_fig(rf=rfwant, nmol=molwant)
vctc_n_fig(rf=rfwant, nmol=2)
vctc_f_fig(rf=rfwant, nmol=2)
clrt_n_fig(rf=rfwant, nmol=2)
clrt_f_fig(rf=rfwant, nmol=2)
chull_n_fig(rf=rfwant, nmol=2)
chull_f_fig(rf=rfwant, nmol=2)
vctc_n_fig(rf=rfwant, nmol=4)
vctc_f_fig(rf=rfwant, nmol=4)
clrt_n_fig(rf=rfwant, nmol=4)
clrt_f_fig(rf=rfwant, nmol=4)
chull_n_fig(rf=rfwant, nmol=4)
chull_f_fig(rf=rfwant, nmol=4)
asph_f_fig(rf=rfwant, nmol=molwant)
rad_n_fig(rf=rfwant, nmol=molwant)
rad_f_fig(rf=rfwant, nmol=molwant,f_want=3)
rad_f_fig(rf=rfwant, nmol=molwant)
form_n_fig(rf=rfwant, nmol=molwant)
form_f_fig(rf=rfwant, nmol=molwant)
cdists_n_fig(rf=rfwant, nmol=2)
cdists_f_fig(rf=rfwant, nmol=2,f_want=3)
cdists_f_fig(rf=rfwant, nmol=2,f_want=6)
cdists_n_fig(rf=rfwant, nmol=4)
cdists_f_fig(rf=rfwant, nmol=4,f_want=3)
cdists_f_fig(rf=rfwant, nmol=4,f_want=6)

R_g_n_fig(rf=rfwant, nmol=molwant,pe=0)
R_g_f_fig(rf=rfwant, nmol=molwant,pe=0)
R_g_nf_fig(rf=rfwant, nmol=molwant,pe=0)
arm_ete_n_fig(rf=rfwant, nmol=molwant,pe=0)
arm_ete_f_fig(rf=rfwant, nmol=molwant,pe=0)
arm_ete_nf_fig(rf=rfwant, nmol=molwant,pe=0)
vctc_n_fig(rf=rfwant, nmol=2,pe=0)
vctc_f_fig(rf=rfwant, nmol=2,pe=0)
clrt_n_fig(rf=rfwant, nmol=2,pe=0)
clrt_f_fig(rf=rfwant, nmol=2,pe=0)
chull_n_fig(rf=rfwant, nmol=2,pe=0)
chull_f_fig(rf=rfwant, nmol=2,pe=0)
vctc_n_fig(rf=rfwant, nmol=4,pe=0)
vctc_f_fig(rf=rfwant, nmol=4,pe=0)
clrt_n_fig(rf=rfwant, nmol=4,pe=0)
clrt_f_fig(rf=rfwant, nmol=4,pe=0)
chull_n_fig(rf=rfwant, nmol=4,pe=0)
chull_f_fig(rf=rfwant, nmol=4,pe=0)
asph_f_fig(rf=rfwant, nmol=molwant,pe=0)
rad_n_fig(rf=rfwant, nmol=molwant,pe=0)
rad_f_fig(rf=rfwant, nmol=molwant,f_want=3,pe=0)
rad_f_fig(rf=rfwant, nmol=molwant,pe=0)
form_n_fig(rf=rfwant, nmol=molwant,pe=0)
form_f_fig(rf=rfwant, nmol=molwant,pe=0)
cdists_n_fig(rf=rfwant, nmol=2,pe=0)
cdists_f_fig(rf=rfwant, nmol=2,f_want=3,pe=0)
cdists_f_fig(rf=rfwant, nmol=2,f_want=6,pe=0)
cdists_n_fig(rf=rfwant, nmol=4,pe=0)
cdists_f_fig(rf=rfwant, nmol=4,f_want=3,pe=0)
cdists_f_fig(rf=rfwant, nmol=4,f_want=6,pe=0)"""

"""msd_n_fig(rf=0, nmol=molwant)
msd_f_fig(rf=0, nmol=molwant)
msdc_n_fig(rf=0, nmol=molwant)
msdc_f_fig(rf=0, nmol=molwant)
rmsd_n_fig(rf=10, nmol=molwant)
rmsd_f_fig(rf=10, nmol=molwant)"""

#plt.show()