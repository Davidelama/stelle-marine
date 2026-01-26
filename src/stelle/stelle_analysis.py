import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import subprocess


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