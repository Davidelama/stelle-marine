import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import subprocess
import math

class MarStatProp:
    """
    Compute and store static properties of a daisy
    cm_mod: how the center of mass for the gyration radius is calculated
    cm =: 0->center of core, cm =: 1->geometric cm ,cm =: 2->weighted cm
    """

    def __init__(self, cgtraj, details, cm_mod=0, n_bins=50):
        self.cgtraj = cgtraj.loc[cgtraj["type"]!=2].copy()
        self.details = details
        self.hlim = self.details["n_beads"]
        self.cm_mod = cm_mod
        self.n_bins = n_bins
        if self.cm_mod==2:
            self.cgtraj.loc[:,"weight"] = np.ones((self.cgtraj.shape[0]))
            self.cgtraj.loc[self.cgtraj['type'] == 1, 'weight'] = (self.details["r_core"] * 2) ** 3
        self.cgtraj.loc[:,"sqdist"] = self.cgtraj.groupby(["timestep", "mol_id"], group_keys=False).apply(cm_dist,center=self.cm_mod)
        self.cgtraj.loc[:,"sqrad"] = self.cgtraj.groupby(["timestep", "mol_id"], group_keys=False).apply(cm_dist, center=0)
        self.cgtraj = arm_definer(self.cgtraj, self.details).copy()
        self._gyration = None
        self._arm_ete = None
        self._asphericity = None
        self._rad_density = None
        self._form = None
        self.binning = None
        self.dai_data = pd.merge(pd.merge(pd.merge(pd.merge(self.gyration, self.asphericity,on=["timestep", "mol_id"]), self.rad_density, on=["timestep", "mol_id"]), self.arm_ete, on=["timestep", "mol_id"]), self.form, on=["timestep", "mol_id"])
        self.dai_data.binning = None
        self.dai_data.binning = self.binning

    @property
    def gyration(self):
        if self._gyration is None:
            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            weighted=False
            if self.cm_mod>1:
                weighted=True
            self._gyration = all_mols.apply(gyr_func, weighted=weighted)
            self._gyration.rename("gyration", inplace=True)
        return self._gyration

    @property
    def arm_gyration(self):
        if self._arm_gyration is None:
            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            self._arm_gyration = all_mols.apply(arm_gyr_func)
            self._arm_gyration.rename("arm_gyration", inplace=True)
        return self._arm_gyration

    @property
    def arm_ete(self):
        if self._arm_ete is None:
            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            self._arm_ete = all_mols.apply(arm_ete_func)
            self._arm_ete.rename("arm_ete", inplace=True)
        return self._arm_ete

    @property
    def asphericity(self):
        if self._asphericity is None:
            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            self._asphericity = all_mols.apply(asph, center=self.cm_mod)
            self._asphericity.rename("asphericity", inplace=True)
        return self._asphericity

    @property
    def rad_density(self):
        if self._rad_density is None:
            self.cgtraj["dist"] = self.cgtraj.sqrad.apply(np.sqrt)
            dmin = self.details["r_core"] / 2 + .5
            dmax = min(self.cgtraj.dist.max(), np.sqrt(self.hlim * 20) + dmin)

            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            self._rad_density = all_mols.dist.apply(function_hist, dmin, dmax, self.n_bins)
            self._rad_density.rename("rad_density", inplace=True)
            self.binning = pd.DataFrame(
                {"rad_density": np.linspace(dmin, dmax, self.n_bins) + (dmax - dmin) / (self.n_bins * 2)})
            #plt.plot(self.binning,self._rad_density.mean())
            #plt.show()
        return self._rad_density

    @property
    def form(self):
        if self._form is None:
            all_mols = self.cgtraj.groupby(["timestep", "mol_id"])
            self._form = all_mols.apply(form_func)
            self._form.rename("form_factor", inplace=True)
        return self._form

class MarDynProp:
    """
    Compute and store static properties of a daisy
    cm_mod: how the center of mass for the gyration radius is calculated
    cm =: 0->center of core, cm =: 1->geometric cm ,cm =: 2->weighted cm
    """

    def __init__(self, msd_traj, ang_traj, details):
        self.msd_traj = msd_traj.copy()
        self.ang_traj = ang_traj.copy()
        self.details = details
        self.msd_data = msd_func(self.msd_traj.copy(), self.details)
        self.rmsd_data = rmsd_func(self.msd_traj, self.ang_traj, self.details)
        self.dyn_data = pd.merge(self.msd_data, self.rmsd_data, on=["time"])


def gyr_func(grp, weighted=False):
    if weighted:
        avg = np.sqrt((grp.sqdist * grp.weight).sum() / grp.weight.sum())
    else:
        avg = np.sqrt(grp.sqdist.mean())
    return (avg)  #we "forget" that the core colloid has a huge mass - talk about it with Luca?

def arm_gyr_func(grp):
    arm_sqdist=grp.loc[grp["arm"]!=0].groupby(["arm"], group_keys=False).apply(cm_dist,center=1)
    return (np.sqrt(arm_sqdist.mean().mean()))

def arm_ete_func(grp):
    arm_ete=grp.loc[grp["arm"]!=0].groupby(["arm"], group_keys=False).apply(ete)
    return (np.sqrt(arm_ete.mean().mean()))

def form_func(grp):
    vals=50 #50 was the original value
    qrand=30
    qvals=np.logspace(-2,3,num=vals)

    qall = np.random.rand(vals,qrand,2)
    qnorm = np.linalg.norm(qall, axis=2)
    qall = qall/qnorm[:,:,None]*qvals[:,None,None]


    ctr = grp.loc[grp.type == 1, ["x", "y"]].values[0]
    sub = grp[["x", "y"]].values - ctr


    form=np.zeros((vals,qrand))
    for i in np.arange(vals):
        for j in np.arange(qrand):
            form[i, j] = np.abs(np.mean(np.exp(1j * np.dot(sub,qall[i,j])))) ** 2
    return np.mean(form,axis=1)

def cm_dist(grp, center=0):
    if center == 0:
        ctr = grp.loc[grp['at_id'] == 1].head(1)[["x", "y"]]
    elif center == 1:
        ctr = grp.mean()[["x", "y"]]
    else:
        ctr = np.average(grp[["x", "y"]], weights=grp["weight"], axis=0)

    sub = grp[["x", "y"]] - ctr
    grp["sqdist"] = sub.x ** 2 + sub.y ** 2
    return (grp["sqdist"])

def ete(grp):
    first = grp.loc[grp['at_id'] == grp['at_id'].min()][["x", "y"]]
    last = grp.loc[grp['at_id'] == grp['at_id'].max()][["x", "y"]]
    return np.sqrt((first.x-last.x)**2 + (first.y-last.y)**2)

def asph(grp,center=0, pbc_L=0):
    if center == 0:
        ctr = grp.loc[grp.type == 1].head(1)[["x", "y"]]
    elif center == 1:
        ctr = grp.mean()[["x", "y"]]
    else:
        ctr = np.average(grp[["x", "y"]], weights=grp["weight"], axis=0)
    sub = grp[["x", "y"]] - ctr
    if pbc_L>0:
        sub -= np.rint(sub / pbc_L) * pbc_L

    pos = np.array([sub.x.values,sub.y.values])

    gyr_tens = np.array([[np.mean(pos[i]*pos[j]) for i in range(2)] for j in range(2)])
    EV , evec = np.linalg.eig(gyr_tens)
    axis=np.sqrt(EV)
    A=((axis[0]-axis[1])**2)/(np.sum(axis)**2)
    return A

def function_hist(seq, ini, final, nbins):
    vector = seq.values
    bins = np.linspace(ini, final, nbins + 1)
    hist = np.histogram(np.array(vector), bins, range=(ini, final))
    return hist[0]

def arm_definer(cgtraj, details):
    Ntot=1+details["n_beads"]*details["functionality"]
    cgtraj.loc[:,"arm"] = (((cgtraj.at_id-1)%Ntot - 1) // details["n_beads"] + 1)+details["functionality"]*(cgtraj.mol_id-1)
    cgtraj.loc[cgtraj.type == 1, "arm"] = 0
    return cgtraj

def rmsd_func(traj, traj_unwrap, details):
    if details['brownian'] == 0:
        dt = min(0.001, 0.01 / details["gamma"])
    else:
        dt = 0.0001

    traj_unwrap["theta"] = np.arctan2(traj_unwrap["muy"], traj_unwrap["mux"])

    traj_unwrap.reset_index(level='timestep', inplace=True)
    times_unwrap = pd.unique(traj_unwrap['timestep'])
    trajlistunwrap = np.array([data["theta"] for _, data in traj_unwrap.groupby(["timestep"])])
    # plt.plot(times_unwrap,trajlistunwrap[:,0]/np.pi,'b',ls="--")
    trajlistunwrap = np.unwrap(trajlistunwrap, axis=0)
    # plt.plot(times_unwrap,trajlistunwrap[:,0]/np.pi,'r',ls="--")


    traj["theta"] = np.arctan2(traj["muy"], traj["mux"])
    tmax = traj.index.max()[1]
    nmax = int(np.log2(tmax)) + 1

    frames = 2 ** np.arange(nmax)

    msd_c = np.zeros(nmax)
    msd_c_std = np.zeros(nmax)

    traj.reset_index(level='timestep', inplace=True)
    times = pd.unique(traj['timestep'])
    unwrap_idx = np.zeros(len(times))
    trajlistc = np.array([data[data.type == 1]["theta"] for _, data in traj.groupby(["timestep"])])
    # plt.plot(times,np.unwrap(trajlistc[:,0])/np.pi,'b',lw=2)
    for i, time in enumerate(times):
        idx = find_nearest(times_unwrap, time)
        trajlistc[i] += np.rint((trajlistunwrap[idx] - trajlistc[i]) / (2 * np.pi)) * np.pi * 2

    maxrot=np.max(np.abs(trajlistunwrap[1:,0]-trajlistunwrap[:-1,0])/np.pi)
    if maxrot > 0.5:
        print(f"Dangerously large core rotations: {maxrot}")
    # plt.plot(times,trajlistc[:,0]/np.pi,'r',lw=2)
    # plt.show()
    # plt.plot(times_unwrap[1:],np.abs(trajlistunwrap[1:,0]-trajlistunwrap[:-1,0])/np.pi,'g')
    # plt.show()
    utr, utc = np.triu_indices(len(times), 1)
    diffs = np.abs(times[utr] - times[utc])

    utr, utc = np.triu_indices(len(times), 1)
    diffs = np.abs(times[utr] - times[utc])

    for i, tval in enumerate(frames):
        cond = diffs == tval
        nvals = min(np.sum(cond), tmax // tval)

        msdvalc = (trajlistc[utr[cond], :] - trajlistc[utc[cond], :]) ** 2
        msd_c[i] = np.mean(msdvalc)
        msd_c_std[i] = np.std(msdvalc) / np.sqrt(nvals)
        # msd[i]=np.mean((trajlist[utr[cond],1]-trajlist[utc[cond],1])**2+(trajlist[utr[cond],2]-trajlist[utc[cond],2])**2)

    t = frames * dt
    results = pd.DataFrame(columns=["time", "angle_msd_core", "angle_msd_core_std"], data=np.c_[t, msd_c, msd_c_std])
    return results


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def msd_func(traj, details):
    diam = 15
    if details['brownian'] == 0:
        dt = min(0.001, 0.01 / details["gamma"])
    else:
        dt = 0.0001

    traj[["x", "y"]] *= diam
    traj["theta"] = np.arctan2(traj["muy"], traj["mux"])
    # plt.hist(traj.theta, range=(-np.pi, np.pi), bins=np.linspace(-np.pi, np.pi, 20))
    # plt.show()

    tmax = traj.index.max()[1]
    nmax = int(np.log2(tmax)) + 1

    frames = 2 ** np.arange(nmax)
    msd = np.zeros(nmax)
    msd_std = np.zeros(nmax)

    msd_c = np.zeros(nmax)
    msd_c_std = np.zeros(nmax)

    traj.reset_index(level='timestep', inplace=True)
    times = pd.unique(traj['timestep'])
    trajlist = np.array([data[data.type != 2][["x", "y"]] for _, data in traj.groupby(["timestep"])])
    trajlistc = np.array([data[data.type == 1][["x", "y"]] for _, data in traj.groupby(["timestep"])])
    utr, utc = np.triu_indices(len(times), 1)
    diffs = np.abs(times[utr] - times[utc])

    for i, tval in enumerate(frames):
        cond = diffs == tval
        nvals = min(np.sum(cond), tmax // tval)
        msdval = np.sum((trajlist[utr[cond], :, :] - trajlist[utc[cond], :, :]) ** 2, axis=2)
        msd[i] = np.mean(msdval)
        msd_std[i] = np.std(msdval) / np.sqrt(nvals)

        msdvalc = np.sum((trajlistc[utr[cond], :, :] - trajlistc[utc[cond], :, :]) ** 2, axis=2)
        msd_c[i] = np.mean(msdvalc)
        msd_c_std[i] = np.std(msdvalc) / np.sqrt(nvals)
        # msd[i]=np.mean((trajlist[utr[cond],1]-trajlist[utc[cond],1])**2+(trajlist[utr[cond],2]-trajlist[utc[cond],2])**2)

    t = frames * dt
    results = pd.DataFrame(columns=["time", "msd_tot", "msd_tot_std", "msd_core", "msd_core_std"],
                               data=np.c_[t, msd, msd_std, msd_c, msd_c_std])

    return results
