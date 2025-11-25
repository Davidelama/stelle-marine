"""
Implements classes and functions to build daisy-mers, create lammps simulation files.
"""

import os
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from shutil import copy2
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timedelta
from IO import lammpsdat_reader
from IO import  get_details
from IO import get_name
from itertools import combinations_with_replacement

builderdir = Path(os.path.dirname(os.path.abspath(__file__)))


class Single:
    def __init__(self, details: dict):#r_core: float, graft: int, star: int, implosion: int):
        """Simple class implementing a daisy.

        Parameters
        ----------
        system: pd.DataFrame
            data about the system. Must contain columns: 'x','y','z', 'mol_id', 'at_id', 'at_type'...
        """
        self.data = pd.DataFrame([[1,1,0,0,0]], index=[1], columns=['at_idx','at_type', 'x', 'y', 'z'])
        self.details = details
        self.ellipsoids = add_ellipsoids(self.data, self.details, flat=True)
        self._name = None
        
    @property
    def name(self):
        if self._name is None:
            self._name=get_name(self.details)
        return self._name

class PBSParams:
    def __init__(self, scriptname="single.builder", name='sim', fin='in.lmp', ncpus=8, queue='compphys_cpuQ', hours=23, minutes=50, delta=120):
        """PBS job submitter

        Parameters
        ----------
        n_cpus : int
           number of cpus requested
        hours : int
            number of hours requested
        minutes : int
            minutes requested in addition to the hours
        queue : str
            name of queue requested
        name : str
            name of job
        delta : float
            safety time before end of simulation
        """
        if queue == 'compphys_cpuQ':
            hours = 23
            minutes = 50
        elif queue == 'short_cpuQ':
            hours = 5
            minutes = 50
        elif queue == 'topol_cpuQ':
            hours = 71#47
            minutes = 50
        trun = timedelta(hours=hours, minutes=minutes)
        tbuffer = timedelta(minutes=delta)
        tsim = trun - tbuffer
        self.data = {'n_cpus': ncpus,
                     'n_mpi': ncpus,
                     'name': name,
                     'queue': queue,
                     'trun': self.__timedelta_fmt(trun),
                     'tsim': self.__timedelta_fmt(tsim),
                     'datafile': fin,
                     'scriptname': scriptname}

    @staticmethod
    def __timedelta_fmt(dt):
        hours = dt.days * 24 + dt.seconds // 3600
        minutes = dt.seconds % 3600 // 60
        seconds = dt.seconds % 60
        return f'{hours}:{minutes:02d}:{seconds:02d}'

class SLURMParams:
    def __init__(self, scriptname="single.builder", name='sim', nnodes=1, partition='boost_usr_prod', project='INF25_biophys', qos='normal',  fin='in.lmp', hours=23, minutes=50, delta=120):
        """PBS job submitter

        Parameters
        ----------
        n_cpus : int
           number of cpus requested
        hours : int
            number of hours requested
        minutes : int
            minutes requested in addition to the hours
        queue : str
            name of queue requested
        name : str
            name of job
        delta : float
            safety time before end of simulation
        """
        if qos == 'normal':
            hours = 23
            minutes = 50
        elif qos == 'boost_qos_lprod':
            hours = 96
            minutes = 50
        trun = timedelta(hours=hours, minutes=minutes)
        tbuffer = timedelta(minutes=delta)
        tsim = trun - tbuffer
        self.data = {'n_cpus': nnodes*32,
                     'nnodes': nnodes,
                     'name': name,
                     'project': project,
                     'trun': self.__timedelta_fmt(trun),
                     'tsim': self.__timedelta_fmt(tsim),
                     'datafile': fin,
                     'qos': qos,
                     'partition': partition,
                     'scriptname': scriptname}

    @staticmethod
    def __timedelta_fmt(dt):
        hours = dt.days * 24 + dt.seconds // 3600
        minutes = dt.seconds % 3600 // 60
        seconds = dt.seconds % 60
        return f'{hours}:{minutes:02d}:{seconds:02d}'

class SimulationRenderer:
    def __init__(self, lmpdata, lmpinput, pbspar, slurmpar, n_restarts=10, delta_box=10, suffix=''):
        cwd = Path.cwd()

        self.suffix = suffix
        self.name = lmpdata.single.name
        self.data = {'lmpdata': lmpdata.lammps_data,
                     'lmpdump': lmpdata.lammps_data,
                     'lmpinput': vars(lmpinput),
                     'pbshead': self._create_pbs_data(self.name, pbspar.data),
                     'pbsres': self._create_pbs_data(self.name, pbspar.data),
                     'shead': self._create_slurm_data(self.name, slurmpar.data),
                     'sres': self._create_slurm_data(self.name, slurmpar.data),
                     'launcher': self._create_launcher(n_restarts),
                     'launcher_restart': self._create_launcher(n_restarts),
                     'slauncher': self._create_launcher(n_restarts),
                     'slauncher_restart': self._create_launcher(n_restarts)}
        self.template = {'lmpdata': LammpsDatafile.template_lmp_data,
                         'lmpdump': lmpdata.template_dump_input,
                         'lmpinput': LammpsLangevinInput.template_lmp_input,
                         'pbshead': 'head.pbs',
                         'pbsres': 'restart.pbs',
                         'shead': 'shead.slrm',
                         'sres': 'srestart.slrm',
                         'launcher': 'chain.sh',
                         'launcher_restart': 'chain_restart.sh',
                         'slauncher': 'schain.sh',
                         'slauncher_restart': 'schain_restart.sh'}
        self.outfile = {'lmpdata': self.name + '.dat',
                        'lmpdump': self.name +'.dump',
                        'lmpinput': self.name + '.lmp',
                        'pbshead': 'head.pbs',
                        'pbsres': 'restart.pbs',
                        'shead': 'shead.slrm',
                        'sres': 'srestart.slrm',
                        'launcher': 'chain.sh',
                        'launcher_restart': 'chain_restart.sh',
                        'slauncher': 'schain.sh',
                        'slauncher_restart': 'schain_restart.sh'}
        self.template_dir = builderdir / 'templates'
        self.lammpsinc_dir = self.template_dir / 'lammps_includes'
        
    @staticmethod
    def _create_pbs_data(name, pbspars):
        pbs = copy(pbspars)
        pbs['name'] = name
        pbs['datafile'] = name + '.dat'
        pbs['input'] = name + '.lmp'
        pbs['date'] = datetime.today().strftime('%Y-%m-%d')
        return pbs

    @staticmethod
    def _create_slurm_data(name, slurmpars):
        slurm = copy(slurmpars)
        slurm['name'] = name
        slurm['datafile'] = name + '.dat'
        slurm['input'] = name + '.lmp'
        slurm['date'] = datetime.today().strftime('%Y-%m-%d')
        return slurm


    @staticmethod
    def _create_launcher(n_restarts):
        params = {'date': datetime.today().strftime('%Y-%m-%d'),
                  'nrestarts': n_restarts}
        return params

    @staticmethod
    def _render(template_dir, template, data, outfile):
        file_loader = FileSystemLoader(template_dir)
        env = Environment(loader=file_loader, undefined=StrictUndefined)
        with open(outfile, 'w') as f:
            template = env.get_template(template)
            f.write(template.render(**data))

    def render_sim(self, outpath, exist_ok=False):
        outdir = outpath / (self.name + self.suffix)
        outdir.mkdir(exist_ok=exist_ok)
        for k, tmpl in self.template.items():
            self._render(self.template_dir, tmpl, self.data[k], outdir / self.outfile[k])
        for file in list(self.lammpsinc_dir.glob('*.lmp')):
            copy2(file, outdir)

class LammpsLangevinInput:
    template_dir = builderdir / 'templates'
    template_lmp_input = 'template_single.lmp'

    def __init__(self, single, scriptname="single.builder", n_thermo=1e5, timestep=.001, bending=1, runtime=1e9, restime=1e7, dumptime=1e6, balancetime=1e6, temp=1.0, tau_damp=1.0): # Delta_cm=3.5, Delta_cc=8.,
        """Lammps data for a Langevin simulation

        Parameters
        ----------
        thermalization: np.bool
            True if this simulation is for thermalization False otherwise
        bending : float
           bending  constant for cosine angle interaction
        steps : int
            simulation steps
        n_beads: int
            number of beads per petal
        functionality: int
            number of petals
        r_core: float
            radius of core
        grafted: np.bool
            If True, grafts the arm contact points to the core
        n_thermo : int
            print thermo every n_thermo steps
        n_balance : int
            balance processors every n_balance steps
        n_dump : int
            dump every n_dump steps
        temp : float
            Temperature of the simulation (LJ units)
        tau_damp : float
            damping for Langevine Thermostat
        seed : int
            PRNG seed
        
        """
        self.timestep = timestep
        self.runtime = int(runtime)
        self.details = single.details
        self.conf = int(self.details["r_conf"]>0)
        self.peclet = self.details["peclet"]
        self.conf_radius = self.details["r_conf"]
        self.balancetime = int(balancetime)
        self.n_thermo = int(n_thermo)
        self.restime = int(restime)
        self.dumptime = int(dumptime)
        self.temp = temp
        self.tau_damp = tau_damp
        self.rcutoff = 1.12246152962189
        self.equitime = 1e5
        self.scriptname = scriptname
        self.brownian = int(self.details["brownian"])
        self.Dr = self.details["Dr"]
        
        self.name = single.name
        date = datetime.now()
        self.date = date.strftime("%m/%d/%Y, %H:%M:%S")
        
        seed=self.details["seed_start"]
        
        if seed is None:
            seed = int.from_bytes(os.urandom(3), 'little')
        self.seed = seed
        if self.seed < 0:
            raise ValueError('Seed must be >=0')
        if self.tau_damp <= 0:
            raise ValueError('taudamp must be >0')
        self.jinja_loader = FileSystemLoader(LammpsLangevinInput.template_dir)
        self.jinja_env = Environment(loader=self.jinja_loader, undefined=StrictUndefined)
        self.jinja_template = self.jinja_env.get_template(LammpsLangevinInput.template_lmp_input)

    def render(self):
        return self.jinja_template.render(**vars(self))

    def to_file(self, outfile):
        """Generate Lammps input file 'outfile'

        Parameters
        ----------
        outfile : str
            Filename (path) of the file to write.
        """
        with open(outfile, 'w') as f:
            f.write(self.render())
            

class LammpsDatafile:
    template_dir = builderdir / 'templates'
    template_lmp_data = 'template_lmp_input.dat'
    template_dump_input = 'template_dump_input.dump'

    def __init__(self, single: Single, scriptname="single.builder", delta_box=10):
        """Write Lammps input files to simulate a daisy system.

        Parameters
        ----------
        daisy : Daisy
            The daisy system to simulate
        scriptname : str
            The name of the script generating the lammps files
        delta_box: float
            Safety distance between maximum extent of polymer and box sides
        """
        self.single = single
        self.scriptname = scriptname
        self._box = LammpsDatafile.box_builder(self.single, delta=delta_box)
        self._lammps_data = None

    @property
    def box(self):
        return self._box

    @property
    def lammps_data(self):
        if self._lammps_data is None:
            self._lammps_data = self._set_lammps_data()
        return self._lammps_data

    @staticmethod
    def box_builder(single: Single, delta=10):
        """Build a cubic box around a daisy.

        Parameters
        ----------
        daisy : Daisy
            system to be contained in the box
        delta : float
            extra space for each side (in units of sigma)

        Returns
        -------
        np.array
            box sizes [[-dx,dx], [-dy,dy], [-dz,dz]]
        """
        dx = delta + single.details["r_conf"]
        return np.array([[-dx, dx], [-dx, dx], [-.5, .5]])
    
    
    def _set_lammps_data(self) -> dict:
        """Generates dictionary to fill datafile jinja2 template

        Returns
        -------
        Dict
            Dictionary passed to jinja2 to write the data file
        """
        system = self.single.data
        system["rmass"]=1.9098593171027443
        system["ellipse"] = 1

        max_at_types = max(system.at_type)
        
        def rower(row):
            atom_string = "{:7d} {:6d} {:6d} {:16.9f} {:16.9f} {:16.9f} {:16.9f}"
            #print(*row[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z']])
            return atom_string.format(*row[['at_idx', 'at_type', 'ellipse', 'rmass', 'x', 'y', 'z']])
        
        atoms_lines=system.astype('O').apply(rower,axis=1)
        
        
        #atoms_lines = system.apply(
        #    lambda row: atom_string.format(*row[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z']]), axis=1)
        
        
        
        #atoms_lines = "{0:7d} {1:6d} {2:6d} {3:16.9f} {4:16.9f} {5:16.9f}".format(system.at_idx, system.mol_idx, system.at_type, system.x, system.y, system.z)

        ellipsoids = self.single.ellipsoids

        ellipsoids_lines = ellipsoids.e_idx.astype(str) + ' ' + ellipsoids.diamx.astype(str) + ' ' + ellipsoids.diamy.astype(str) + ' ' + ellipsoids.diamz.astype(str) + ' ' + ellipsoids.q1.astype(str) + ' ' + ellipsoids.q2.astype(str) + ' ' + ellipsoids.q3.astype(str) + ' ' + ellipsoids.q4.astype(str)

        lammps_data = {'script': self.scriptname, 'date': datetime.today().strftime('%Y-%m-%d'),
                       'n_atoms': system.shape[0],
                       'n_atype': max(system.at_type),
                       'xlo': self.box[0][0], 'xhi': self.box[0][1],
                       'ylo': self.box[1][0], 'yhi': self.box[1][1],
                       'zlo': self.box[2][0], 'zhi': self.box[2][1],
                       'atoms': atoms_lines,
                       'ellipsoids': ellipsoids_lines}
        return lammps_data

    def to_file(self, outfile):
        """Generate Lammps datafile 'outfile'

        Parameters
        ----------
        outfile : str
            Filename (path) of the file to write.
        """
        file_loader = FileSystemLoader(LammpsDatafile.template_dir)
        env = Environment(loader=file_loader, undefined=StrictUndefined)
        template = env.get_template(LammpsDatafile.template_lmp_data)
        dump_template = env.get_template(LammpsDatafile.template_dump_input)
        with open(outfile, 'w') as f:
            f.write(template.render(**self.lammps_data))


def add_ellipsoids(sin_in, details, flat=False):
    npart=len(sin_in)
    single_ellipsoids=pd.DataFrame({})
    id=np.arange(1,npart+1)
    diam=np.ones(npart)
    single_ellipsoids["e_idx"] = id
    single_ellipsoids["diamx"] = diam
    single_ellipsoids["diamy"] = diam
    single_ellipsoids["diamz"] = diam

    if flat:
        theta=np.random.rand(npart)*2*np.pi
        single_ellipsoids["q1"] = np.cos(theta/2)
        single_ellipsoids["q2"] = np.zeros(npart)
        single_ellipsoids["q3"] = np.zeros(npart)
        single_ellipsoids["q4"] = np.sin(theta/2)
    else:
        quatpar=np.random.rand(3,npart)
        u=quatpar[0,:]
        v=quatpar[1,:]
        w=quatpar[2,:]

        single_ellipsoids["q1"] = np.sqrt(1 - u) * np.sin(2 * np.pi * v)
        single_ellipsoids["q2"] = np.sqrt(1 - u) * np.cos(2 * np.pi * v)
        single_ellipsoids["q3"] = np.sqrt(u) * np.sin(2 * np.pi * w)
        single_ellipsoids["q4"] = np.sqrt(u) * np.cos(2 * np.pi * w)

    return single_ellipsoids


def read_single(file):
    system, system_bonds, system_angles, xlim, ylim, zlim = lammpsdat_reader(file)

    system.rename(columns={"at_id": "at_idx", "mol_id": "mol_idx"},inplace=True)

    system_bonds['bonds'] = system_bonds.bd_1.astype(str) + ' ' + system_bonds.bd_2.astype(str)
    system_bonds.rename(columns={"bd_id": "bond_idx", "bd_type": "bond_type"},inplace=True)
    system_bonds.drop(columns=['bd_1', 'bd_2'], inplace=True)

    system_angles['angles'] = system_angles.an_1.astype(str) + ' ' + system_angles.an_2.astype(str) + ' ' + system_angles.an_3.astype(str)
    system_angles.rename(columns={"an_id": "angle_idx", "an_type": "angle_type"},inplace=True)
    system_angles.drop(columns=['an_1', 'an_2', 'an_3'], inplace=True)#not final

    return Single(get_details(os.path.basename(os.path.normpath(Path(file).parents[0])))), xlim

def logtimer(dump, tot,fileout):
    exponent = int(np.log2(dump))
    edump = int(2 ** exponent)
    nedump = int(tot // edump)
    ntot = int(nedump * (exponent + 1) + 1)

    tvec = np.zeros(ntot)
    tarr = 2 ** np.arange(exponent + 1)
    tvec[0:exponent + 1] = tarr
    for i in range(1, nedump):
        tvec[i * (exponent + 1):(i + 1) * (exponent + 1)] = tarr + tvec[i * (exponent + 1) - 1]
    tvec[-1] = tot + 1
    np.savetxt(fileout, tvec, fmt='%d')

#scp -r clicks.tar.gz davide.breoni@hpc2.unitn.it:~/single/src