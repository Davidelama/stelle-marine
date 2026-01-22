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


class Daisy:
    def __init__(self, system: pd.DataFrame, system_bonds: pd.DataFrame, system_angles: pd.DataFrame, system_ellipsoids: pd.DataFrame, details: dict):#r_core: float, graft: int, star: int, implosion: int):
        """Simple class implementing a daisy.

        Parameters
        ----------
        system: pd.DataFrame
            data about the system. Must contain columns: 'x','y','z', 'mol_id', 'at_id', 'at_type'...
        """
        self.data = system
        self.bonds = system_bonds
        self.angles = system_angles
        self.ellipsoids = system_ellipsoids
        self.details = details
        #self.n = int(system.groupby('p_idx')['at_idx'].count().max())    #at_idx = atom index
        #self.f = system.p_idx.nunique()-1
        #self.r_core = r_core
        self.m_core = (self.details["r_core"]/.5)**3
        self.lmol = int(np.ceil(self.details["n_mol"]**(1/2)))
        self.L = (self.details["r_cbond"]+ (self.details["n_beads"]+1) * self.details["r_bond"]) * 2 * self.lmol
        self._name = None
        
    @property
    def name(self):
        if self._name is None:
            self._name=get_name(self.details)
        return self._name

    """
    @property
    
    def rc(self):
        
        Returns
        -------
        float
            radius of the catenane
        
        if self._rc is None:
            coords = self.data[['x', 'y', 'z']].values
            opp_coords = np.roll(coords, -coords.shape[0] // 2, axis=0)
            diam = np.linalg.norm(coords - opp_coords, axis=1).max()
            self._rc = diam / 2
        else:
            return self._rc
    """

class DaisyBuilder:
    def __init__(self, details: dict):#n_beads: int, functionality: int, r_core: float, graft: int, star: int, implosion: int):
        """Class to build stelle with given number of arms, monomers and core radius.

        Parameters
        ----------
        n_beads : int
            number of beads per petal
        functionality : int
            number of petals
        r_core: float
            radius of core in units of monomer diameter sigma
        """
        self.details = details
        self.m_core = (self.details["r_core"]/.5)**3
        if self.details["r_int"] > 0:
            self.details["n_mol"] = 2
            self.details["ghost"] = 0
            self.details["r_conf"] = 0

    def build_system(self) -> Daisy:
        """Build a daisy with n beads per petal, f petals and core radius r_core

        Returns
        -------
        Daisy
            a Daisy class with n beads per petal, f petals and core radius r_core.
        """

        n_tot = 0
        nb_tot = 0
        na_tot = 0
        n_mol = self.details["n_mol"]
        system = pd.DataFrame({})
        system_bonds = pd.DataFrame({})
        system_angles = pd.DataFrame({})

        lmols=int(np.ceil(n_mol**(1/2)))
        self.L=(self.details["r_cbond"]+ (self.details["n_beads"]+1) * self.details["r_bond"]) * 2 * int(np.ceil(self.details["n_mol"]**(1/2)))
        lcell=self.L/lmols
        o=0
        p=0

        for i in range(n_mol):
            cent = [(lcell-self.L)*.5 + lcell*o,(lcell-self.L)*.5 + lcell*p,0]
            if o == lmols-1:
                o = 0
                if p == lmols - 1:
                    p = 0
                else:
                    p += 1
            else:
                o += 1

            star = build_daisy(self.details, mol=i, last=n_tot, center=cent)
            bonds, angles = add_topology(star, self.details, last=n_tot, last_b=nb_tot, last_a=na_tot)

            system = pd.concat([system, star])
            system_bonds = pd.concat([system_bonds, bonds])
            system_angles = pd.concat([system_angles, angles])

            n_tot += len(star)
            nb_tot += len(bonds)
            na_tot += len(angles)

        #system["at_idx"] = np.arange(1, len(system) + 1)
        #system_bonds["at_idx"] = np.arange(1, len(system_bonds) + 1)
        #system_angles["at_idx"] = np.arange(1, len(system_angles) + 1)
        #system.reset_index(drop=True, inplace=True)
        #system_bonds.reset_index(drop=True, inplace=True)
        #system_angles.reset_index(drop=True, inplace=True)

        system_ellipsoids = add_ellipsoids(system, self.details, flat=True)

        return Daisy(system, system_bonds, system_angles, system_ellipsoids, self.details)

class PBSParams:
    def __init__(self, scriptname="stelle.builder", name='sim', fin='in.lmp', ncpus=8, queue='compphysCPUQ', hours=23, minutes=50, delta=120):
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
        if queue == 'compphysCPUQ':
            hours = 23
            minutes = 50
        elif queue == 'shortCPUQ':
            hours = 5
            minutes = 50
        elif queue == 'topolCPUQ':
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
    def __init__(self, scriptname="stelle.builder", name='sim', nnodes=1, partition='boost_usr_prod', project='INF25_biophys', qos='normal',  fin='in.lmp', hours=23, minutes=50, delta=120):
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
        self.name = lmpdata.daisy.name
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
    template_lmp_input = 'template_stelle.lmp'

    def __init__(self, daisy, scriptname="stelle.builder", timestep=.001, bending=0, runtime=1e9, restime=1e7, dumptime=1e6, balancetime=1e6, dumptime_angle=1e4): # Delta_cm=3.5, Delta_cc=8.,
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
        mass=1
        radius=.5
        inertia=2/5*mass*radius**2
        self.timestep = timestep
        self.runtime = int(runtime)
        self.details = daisy.details
        self.n_beadseff = int(self.details["n_beads"])+1
        self.n_mol = int(self.details["n_mol"])
        self.ghost = int(self.details["ghost"])
        self.conf = int(self.details["r_conf"]>0)
        self.start_radius = daisy.L*np.sqrt(2)/2
        self.end_radius = self.details["r_conf"]
        self.n_all = 1+self.n_beadseff*int(self.details["functionality"])
        self.func = int(self.details["functionality"])
        self.balancetime = int(balancetime)
        self.n_thermo =  10*int(1/timestep)
        self.restime = int(restime)
        self.dumptime = int(dumptime)
        self.dumptime_angle = int(dumptime_angle)
        self.rcutoff = 1.12246152962189
        self.equitime = 500000 + self.n_beadseff*50000
        self.Delta_cm = self.details["r_core"] - .5 #r_core+r_monomer-1
        self.Delta_cc = self.details["r_core"] * 2 - 1 #r_core+r_core-1
        self.sigma_core = self.details["r_core"]*2
        self.scriptname = scriptname
        self.brownian = int(self.details["brownian"])
        self.r_cbond = int(self.details["r_cbond"])
        self.r_bond = int(self.details["r_bond"])
        self.tau_damp = 1/self.details["gamma"]
        #all force coefficients must be multiplied by gamma
        self.bending = bending * self.details["gamma"]
        self.peclet = self.details["peclet"] * self.details["gamma"]
        self.epsilon = 10.0 * self.details["gamma"]
        self.temp = self.details["Dt"] * self.details["gamma"]
        self.Tr = self.details["Dr"] * self.details["gamma"]
        if self.brownian == 0:
            self.temp *= mass
            self.Tr *= inertia
        if self.details["Dt"] == 0:
            self.temp = 0.000001
        if self.details["Dr"] == 0:
            self.Tr = 0.000001
        self.contact = int(self.details["contact"])
        self.dtmove = timestep
        self.tmove = 1e5#10*int(1/timestep)
        self.t_force_dump = 1e4
        self.vmove=(daisy.L/daisy.lmol-self.details["r_int"])/(self.tmove*self.dtmove)*.5
        self.int_fix = int(self.details["r_int"]>0)
        self.n_beadseff = int(self.details["n_beads"])+1

        
        self.name = daisy.name  
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

    def __init__(self, daisy: Daisy, scriptname="stelle.builder", delta_box=10):
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
        self.daisy = daisy
        self.scriptname = scriptname
        self._box = LammpsDatafile.box_builder(self.daisy, delta=delta_box)
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
    def box_builder(daisy: Daisy, delta=10):
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
        dx = daisy.L*.5
        if daisy.details["r_conf"]>0:
            dx*=np.sqrt(2)
        return np.array([[-dx, dx], [-dx, dx], [-.5, .5]])
    
    
    def _set_lammps_data(self) -> dict:
        """Generates dictionary to fill datafile jinja2 template

        Returns
        -------
        Dict
            Dictionary passed to jinja2 to write the data file
        """
        system = self.daisy.data
        system["q"] = 0.0
        system["muz"] = 0.0
        theta= np.random.uniform(-np.pi,np.pi,system.shape[0])
        system["mux"] =np.cos(theta)
        system["muy"] = np.sin(theta)
        system["rmass"]=1.9098593171027443
        system["diameter"] = 1.0
        system.loc[system["at_type"] == 2,"diameter"] = 0.001
        system.loc[system["at_type"] == 1, "diameter"] = self.daisy.details["r_core"]*2
        m_pin=0.001
        m_bead=1.0
        if self.daisy.details["r_int"]>0:
            system.loc[system["at_type"] == 1, "rmass"] = 1e12
            system.loc[system["at_type"] == 1, "diameter"] = 0
            Icore = 2/5*self.daisy.m_core*self.daisy.details["r_core"]**2
            m_pin_new = Icore/self.daisy.details["functionality"]/self.daisy.details["r_cbond"]**2
            system.loc[system["at_type"] == 2, "rmass"] = m_pin_new/(1/6*np.pi*0.001**3)
            #m_pin=self.daisy.m_core/self.daisy.details["functionality"]#4*self.daisy.details["r_core"]**2*m_bead/self.daisy.details["functionality"]
            #self.daisy.m_core=1e12
        """
        at_types = len(system.at_type.unique())
        
        mass_lines = system.at_type.unique().astype(str)
        m_array = ['{0:.1f}'.format(self.daisy.m_core)]+["1.0"]*(at_types-1)
        for i in range(at_types):
             mass_lines[i] = mass_lines[i] + ' ' + m_array[i]
        """
        max_at_types = max(system.at_type)
        mass_lines = np.arange(1,max_at_types+1).astype(str)
        m_array = ['{0:.1f}'.format(self.daisy.m_core)] + [str(m_pin)] + [str(m_bead)] * (max_at_types - 2)
        for i in range(max_at_types):
            mass_lines[i] = mass_lines[i] + ' ' + m_array[i]
        
        def rower(row):
            atom_string = "{:7d} {:6d} {:16.9f} {:16.9f} {:16.9f} {:6d} {:16.9f} {:16.9f} {:16.9f} {:16.9f} {:16.9f} {:16.9f}"
            #print(*row[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z']])
            return atom_string.format(*row[['at_idx', 'at_type', 'x', 'y', 'z', 'mol_idx', 'diameter', 'rmass','q','mux','muy','muz']])
        
        atoms_lines=system.astype('O').apply(rower,axis=1)
        
        
        #atoms_lines = system.apply(
        #    lambda row: atom_string.format(*row[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z']]), axis=1)
        
        
        
        #atoms_lines = "{0:7d} {1:6d} {2:6d} {3:16.9f} {4:16.9f} {5:16.9f}".format(system.at_idx, system.mol_idx, system.at_type, system.x, system.y, system.z)
        
        bonds =  self.daisy.bonds 
        angles =  self.daisy.angles
        ellipsoids = self.daisy.ellipsoids
            
        bonds_lines = bonds.bond_idx.astype(str) + ' ' + bonds.bond_type.astype(str) + ' ' + bonds.bonds
        angles_lines = angles.angle_idx.astype(str) + ' ' + angles.angle_type.astype(str) + ' ' + angles.angles
        ellipsoids_lines = ellipsoids.e_idx.astype(str) + ' ' + ellipsoids.diamx.astype(str) + ' ' + ellipsoids.diamy.astype(str) + ' ' + ellipsoids.diamz.astype(str) + ' ' + ellipsoids.q1.astype(str) + ' ' + ellipsoids.q2.astype(str) + ' ' + ellipsoids.q3.astype(str) + ' ' + ellipsoids.q4.astype(str)

        lammps_data = {'script': self.scriptname, 'date': datetime.today().strftime('%Y-%m-%d'),
                       'n_atoms': system.shape[0],
                       'n_bonds': bonds.shape[0],
                       'n_angles': angles.shape[0],
                       'n_ellipsoids': ellipsoids.shape[0],
                       'n_atype': max(system.at_type), 'n_btype': bonds.bond_type.nunique(), 'n_antype': angles.angle_type.nunique(),
                       'xlo': self.box[0][0], 'xhi': self.box[0][1],
                       'ylo': self.box[1][0], 'yhi': self.box[1][1],
                       'zlo': self.box[2][0], 'zhi': self.box[2][1],
                       'masses': mass_lines,
                       'atoms': atoms_lines,
                       'bonds': bonds_lines,
                       'angles': angles_lines,
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

def build_petal(details: dict, theta: float) -> np.array:
    """Build a closed petal along the x axis with `nbeads` points and grafting point at "radius" + 0.5 from the origin and rotated with angles phi (z) and theta (y)."""
    
    nbeadseff = details["n_beads"]+1
    radius = details["r_cbond"]
    rbond = details["r_bond"]

    x=(np.arange(nbeadseff)*rbond + radius)*np.cos(theta)
    y=(np.arange(nbeadseff)*rbond + radius)*np.sin(theta)
    z=np.zeros((nbeadseff))
    
    coor = np.array([x,y,z]).transpose()
    
    return coor


def build_daisy(details: dict, mol=0, last=0, center=[0,0,0]) -> pd.DataFrame: #rc_float
    """
    Build a daisy with "nbeads" beads per petal, "functionality" petals and core radius "r_core"
    """
    nbeadseff=details["n_beads"]+1
    xc=center[0]
    yc=center[1]
    zc=center[2]
    f=details["functionality"]
    phase=2*np.pi/(2*f)
    # grafting angles

    
    # build petals
        
    daisy = pd.DataFrame([[1,0,0,0,0]], index=[1], columns=['at_type', 'p_idx', 'x', 'y', 'z'])
    theta = np.linspace(0,2*np.pi, num=details["functionality"],endpoint=False)+phase
    for f in range(details["functionality"]):
        petal = build_petal(details, theta[f])
        types = np.ones((nbeadseff)) * 3
        types[0]=2
        daisy = pd.concat([daisy, pd.DataFrame(np.column_stack([types,np.ones((nbeadseff))*(f+1),petal]), index=range(2+nbeadseff*f,2+nbeadseff*(f+1)), columns=['at_type', 'p_idx', 'x', 'y', 'z'])])

    daisy = daisy.astype({"at_type": int, "p_idx": int})
    daisy['mol_idx'] = mol+1
    daisy['x'] += xc
    daisy['y'] += yc
    daisy['z'] += zc
    daisy['at_idx'] = daisy.index+last
    daisy = daisy[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z', 'p_idx']]
    return daisy


def add_topology(dai_in: pd.DataFrame, details: dict, last=0, last_b=0, last_a=0) -> pd.DataFrame:
    """
    Creates bonds and angles for a daisy DataFrame
    """
    dai_in = dai_in[dai_in.p_idx != 0]
    daisy_bonds = pd.DataFrame({})
    daisy_angles = pd.DataFrame({})
    #daisy_angles = dai_in.copy()['p_idx']
    nbeads = dai_in.groupby('p_idx').size()

    id_group_bonds = dai_in.groupby('p_idx').head(nbeads.max() - 1).at_idx
    id_group_angles = dai_in.groupby('p_idx').head(nbeads.max() - 2).at_idx

    daisy_bonds['bonds'] = id_group_bonds.astype(str) + ' ' + (id_group_bonds + 1).astype(
        str)  # dai_in.at_idx.astype(str) + ' ' + (dai_in.at_idx + v).astype(str)    #bonds on petals
    daisy_bonds["bond_type"] = 1
    daisy_angles['angles'] = id_group_angles.astype(str) + ' ' + (id_group_angles + 1).astype(str) + ' ' + (
                id_group_angles + 2).astype(str)  # angles on petals
    daisy_angles["angle_type"] = 1
    
    daisy_bonds = daisy_bonds.reset_index().drop(columns='index')
    daisy_angles = daisy_angles.reset_index().drop(columns="index")

    #daisy_bonds.loc[daisy_bonds.index.isin(dai_in.groupby('p_idx').head(1).at_idx.values-2), "bond_type"] = 1
    
    daisy_bonds["bond_idx"]=daisy_bonds.index+1+last_b
    daisy_angles["angle_idx"]=daisy_angles.index+1+last_a
    return daisy_bonds, daisy_angles

def add_ellipsoids(dai_in, details, flat=False):
    econd=dai_in["at_type"]!=2
    if details["r_int"] > 0:
        econd=dai_in["at_type"]==3
    npart=np.sum(econd)
    daisy_ellipsoids=pd.DataFrame({})
    id = dai_in[econd].at_idx.values
    diam=np.ones(npart)
    diam[0]*=details["r_core"]*2 #this is wrong!! - does not consider the other stars
    daisy_ellipsoids["e_idx"] = id
    daisy_ellipsoids["diamx"] = diam
    daisy_ellipsoids["diamy"] = diam
    daisy_ellipsoids["diamz"] = diam

    if flat:
        theta=np.random.rand(npart)*2*np.pi
        daisy_ellipsoids["q1"] = np.cos(theta/2)
        daisy_ellipsoids["q2"] = np.zeros(npart)
        daisy_ellipsoids["q3"] = np.zeros(npart)
        daisy_ellipsoids["q4"] = np.sin(theta/2)
    else:
        quatpar=np.random.rand(3,npart)
        u=quatpar[0,:]
        v=quatpar[1,:]
        w=quatpar[2,:]

        daisy_ellipsoids["q1"] = np.sqrt(1 - u) * np.sin(2 * np.pi * v)
        daisy_ellipsoids["q2"] = np.sqrt(1 - u) * np.cos(2 * np.pi * v)
        daisy_ellipsoids["q3"] = np.sqrt(u) * np.sin(2 * np.pi * w)
        daisy_ellipsoids["q4"] = np.sqrt(u) * np.cos(2 * np.pi * w)

    return daisy_ellipsoids


def read_daisy(file):
    system, system_bonds, system_angles, masses, xlim, ylim, zlim = lammpsdat_reader(file)

    system.rename(columns={"at_id": "at_idx", "mol_id": "mol_idx"},inplace=True)

    system_bonds['bonds'] = system_bonds.bd_1.astype(str) + ' ' + system_bonds.bd_2.astype(str)
    system_bonds.rename(columns={"bd_id": "bond_idx", "bd_type": "bond_type"},inplace=True)
    system_bonds.drop(columns=['bd_1', 'bd_2'], inplace=True)

    system_angles['angles'] = system_angles.an_1.astype(str) + ' ' + system_angles.an_2.astype(str) + ' ' + system_angles.an_3.astype(str)
    system_angles.rename(columns={"an_id": "angle_idx", "an_type": "angle_type"},inplace=True)
    system_angles.drop(columns=['an_1', 'an_2', 'an_3'], inplace=True)

    system_ellipsoids =add_ellipsoids(system,get_details(file),flat=True) #not final

    return Daisy(system, system_bonds, system_angles,system_ellipsoids,get_details(os.path.basename(os.path.normpath(Path(file).parents[0])))), xlim

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

#scp -r clicks.tar.gz davide.breoni@hpc2.unitn.it:~/stelle/src