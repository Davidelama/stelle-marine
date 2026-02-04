# import numpy as np
import re
from pathlib import Path

import pandas as pd
import numpy as np
from stelle.IO import reconstruct_traj
from stelle.IO import get_name
from stelle.IO import get_details
from stelle.stelle_analysis import MarStatProp
from stelle.stelle_analysis import MarDynProp


class AnalysisPipeline:
    """
    Performs the whole analysis of a trajectory starting from lammps data.
    The pipeline proceeds as follows
    1. Coarse-graining of the system
    2. Computing local quantities (per elementary ring, e.g. local twist)
    3. Compute global quantities (per catenane, e.g. Rg2, writhe)
    """

    def __init__(self, input_dir, cm_mod=0):
        """
        Initialize the analysis and in particular the coarse-graining procedure
        Parameters
        ----------
        input_dir: input directory storing the lammps trajectory with .lammpstrj  extension
        cm_mod: how the center of mass for the gyration radius is calculated
        cm =: 0->center of core, cm =: 1->geometric cm ,cm =: 2->weighted cm
        """
        self.simdir = Path(input_dir)
        self.details = AnalysisPipeline.get_details(input_dir)
        self.pref = "mar"
        self._traj = None
        self._msd_traj = None
        self._ang_traj = None
        self._start_conf = None
        self._cat = None
        self._static = None
        self._dynamical = None
        self.cm_mod = cm_mod

    def __str__(self):
        return get_name(details=self.details,prefix=False)

    def __repr__(self):
        return f"AnalysisPipeline({self.simdir})"

    @staticmethod
    def get_details(input_dir):
        input_dir = Path(input_dir)
        return get_details(input_dir.name)

    def __mkoutdir(self, outdir, flag=True):
        outdir = Path(outdir)
        sysname = self.__str__()
        (outdir / sysname).mkdir(parents=True, exist_ok=flag)
        return outdir / sysname

    @property
    def traj(self):
        if self._traj is None:
            self._traj = reconstruct_traj(self.simdir.glob(f'{self.pref}*.dat.lammpstrj'))
        return self._traj

    @property
    def msd_traj(self):
        if self._msd_traj is None:
            self._msd_traj = reconstruct_traj(self.simdir.glob(f'{self.pref}*_msd.lammpstrj'), cols=('at_id', 'type', 'x', 'y','mux','muy'))
        return self._msd_traj

    @property
    def ang_traj(self):
        if self._ang_traj is None:
            self._ang_traj = reconstruct_traj(self.simdir.glob(f'{self.pref}*_angle_unwrap.lammpstrj'), cols=('at_id','mux','muy'))
        return self._ang_traj

    @property
    def start_conf(self):
        if self._start_conf is None:
            self._start_conf = reconstruct_traj(self.simdir.glob(f'{self.pref}*dump'))
        return self._start_conf

    @property
    def cat(self):
        if self._cat is None:
            self._cat = self.cg(self.traj, m_a=self.m_a, verse=self.verse, twist_normals=self.twn)
        return self._cat

    @property
    def static_properties(self):
        if self._static is None:
            self._static = MarStatProp(self.traj, self.details, cm_mod=self.cm_mod)  #self.n_beads*(self.star+1)
            self._static = self._static.dai_data
        return self._static

    @property
    def dynamical_properties(self):
        if self._dynamical is None:
            self._dynamical = MarDynProp(self.msd_traj, self.ang_traj, self.details)  #self.n_beads*(self.star+1)
            self._dynamical = self._dynamical.dyn_data
        return self._dynamical

    def load_static_properties(self, input_dir):
        fname = self.__str__() + '_static_properties.pqt'
        self._static = pd.read_parquet(input_dir / fname)

    def save_static_properties(self, outdir):
        fname = self.__str__() + '_static_properties.pqt'
        bname = self.__str__() + '_static_properties_binning.csv'
        outdir = self.__mkoutdir(outdir)
        stat = self.static_properties
        stat.to_parquet(outdir / fname)
        stat.binning.to_csv(outdir / bname)

    def load_dynamical_properties(self, input_dir):
        fname = self.__str__() + '_dynamical_properties.pqt'
        self._dynamical = pd.read_csv(input_dir / fname)

    def save_dynamical_properties(self, outdir):
        fname = self.__str__() + '_dynamical_properties.csv'
        outdir = self.__mkoutdir(outdir)
        dyn = self.dynamical_properties
        dyn.to_csv(outdir / fname)