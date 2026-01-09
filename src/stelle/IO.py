import numpy as np
import pandas as pd
from copy import copy
import re

def lammpsdump_reader(file, cols=('at_id', 'mol_id', 'type', 'x', 'y', 'z')):
    """
    Reads the LAMMPSDUMP files (LAMMPS Trajectories) and stores the results
    in a pandas DataFrame
    """
    ts = 'TIMESTEP'
    nbr = 'NUMBER'
    crd = 'ATOMS id'
    n_cols = len(cols)
    timesteps = []
    confs = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if ts in line:
                timesteps.append(int(f.readline()))
            if nbr in line:
                n_atoms = int(f.readline())
                for i in range(4):
                    f.readline()
            if crd in line:
                coords = np.zeros((n_atoms, n_cols))
                for i in range(n_atoms):
                    line = f.readline().split()
                    coords[i] = np.array(line)
                confs.append(coords)
            line = f.readline()
    confs = np.array(confs)
    n_atoms = confs.shape[1]
    n_frames = confs.shape[0]
    confs = confs.reshape((confs.shape[0] * confs.shape[1], confs.shape[2]))
    data = pd.DataFrame(confs, columns=cols)
    data[['x', 'y']] = data[['x', 'y']].astype(float)
    if 'z' in cols:
        data[['z']] = data[['z']].astype(float)
    data[['at_id', 'type']] = data[['at_id', 'type']].astype(int)
    if 'mol_id' in cols:
        data[['mol_id']] = data[['mol_id']].astype(int)
    # dt = timesteps[1] - timesteps[0]
    data['timestep'] = np.repeat(np.array(timesteps), n_atoms)
    data['frame'] = np.repeat(np.arange(n_frames), n_atoms)
    data.set_index(['frame', 'timestep'], inplace=True)
    return data[list(cols)]


def lammpsdump_writer(filename: str, data: pd.DataFrame, boxlen):
    """
    Write a lammpsdump file to be visualized in VMD
    """
    assert {'at_id', 'mol_id', 'type', 'x', 'y', 'z'} <= set(data.columns)
    ts_str = "ITEM: TIMESTEP\n{:d}\n"
    n_atn_str = "ITEM: NUMBER OF ATOMS\n{:d}\n"
    boxlen_str = f"ITEM: BOX BOUNDS pp pp pp\n" + 3 * f"{-boxlen:.16e} {boxlen:.16e}\n"
    coord_str = "ITEM: ATOMS id mol type xu yu zu\n"
    at_str = "{:10d} {:6d} {:3d} {:12.6f} {:12.6f} {:12.6f}\n"
    with open(filename, 'w') as fout:
        for ts, new in data.groupby(level='timestep'):
            data = new.droplevel('timestep').reset_index()
            # timestep = ts
            fout.write(ts_str.format(ts))
            fout.write(n_atn_str.format(data.shape[0]))
            fout.write(boxlen_str)
            fout.write(coord_str)
            data.reset_index()
            for index, row in data.iterrows():
                fout.write(at_str.format(int(row['at_id']),int(row['mol_id']),int(row['type']), row['x'], row['y'], row['z']))


def knt_writer(filename: str, data: pd.DataFrame, colx='x', coly='y', colz='z'):
    """
    Write a .knt file to be analyzed with entang
    """
    n_atn_str = "{:10d}\n"
    at_str = "{:16.9f} {:16.9f} {:16.9f}\n"
    with open(filename, 'w') as fout:
        for ts, new in data.groupby(level='timestep'):
            data = new.droplevel('timestep').reset_index()
            fout.write(n_atn_str.format(data.shape[0] + 1))
            data.reset_index()
            for index, row in data.iterrows():
                fout.write(at_str.format(row[colx], row[coly], row[colz]))
            row = data.iloc[0]
            fout.write(at_str.format(row[colx], row[coly], row[colz]))


def reconstruct_traj(filelist, cols=('at_id', 'mol_id', 'type', 'x', 'y', 'z')):
    """
    Collate a set of pandas traj
    files into a unique trajectory, sorting according to timestep and dropping duplicates.

    Parameters
    ----------
    filelist: list
        list of files in lammpstrj format to be read by ror.IO.lammpsdump_reader

    Returns
    -------

    """
    data = [lammpsdump_reader(f, cols=cols) for f in filelist]
    ts = [d.index[-1][1] for d in data]
    data_sorted = [d for _, d in sorted(zip(ts, data))]
    shifts = np.array([d.index[-1][0] for d in data_sorted])
    # shifts -= shifts[0]
    for d, s in zip(data_sorted[1:], shifts[:-1]):
        d.index.set_levels(d.index.levels[0] + s, level='frame', inplace=True)
    traj = pd.concat(data_sorted)
    traj.reset_index(level=1, inplace=True)
    # DO NOT drop duplicates on float values, only on integer/strings
    traj = traj.drop_duplicates(subset=['timestep', 'at_id'])
    traj.set_index(['timestep'], append=True, inplace=True)
    return traj



def to_vtf_file(data, M, n, colx, coly, colz, nx, ny, nz, filename):
    """
        Write a .vtf file to be opened with VMD
        Note: assumes that all frames have the same value of M
    """
    at_str = "{:d} {:16.9f} {:16.9f} {:16.9f}\n"
    s = 0.5 * n / np.pi
    with open(filename, 'w') as fout:
        # write preamble (topology)
        fout.write(f"atom default name CENTER radius 1.0\n")
        max_line_len = 20
        iters = M // max_line_len
        # normals atoms
        for i in range(iters):
            idx0 = i * max_line_len
            fout.write(f"atom {2 * idx0 + 1}")
            for l in range(1, max_line_len):
                idx = idx0 + l
                if idx >= M:
                    break
                fout.write(f',{2 * idx + 1}')
            fout.write(f" name NORMAL radius 0.5\n")
        # bonds
        for i in range(iters):
            idx0 = i * max_line_len
            fout.write(f"bonds {2 * idx0}:{2 * (idx0 + 1)},{2 * idx0}:{2 * idx0 + 1}")
            for l in range(1, max_line_len):
                idx = idx0 + l
                if idx >= M:
                    break
                fout.write(f',{2 * idx}:{2 * (idx + 1)},{2 * idx}:{2 * idx + 1}')
            fout.write(f"\n")
        # write coordinates
        #for ts, new in data.groupby(level=CatCG.conf_id):
        #    data = new.droplevel(CatCG.conf_id).reset_index()
        #    # data.reset_index()
        #    fout.write("timestep indexed\n")
        #    for index, row in data.iterrows():
        #        fout.write(at_str.format(2 * index, row[colx], row[coly], row[colz]))
        #        fout.write(at_str.format(2 * index + 1, row[colx] + s * row[nx], row[coly] + s * row[ny],
        #                                 row[colz] + s * row[nz]))


def lammpsdat_reader(file, type="angles"):
    """
    Reads the LAMMPSDUMP files (LAMMPS Trajectories) and stores the results
    in a pandas DataFrame
    """
    at_cols=[]
    bd_cols=[]
    an_cols=[]
    masses=[]
    xlim = np.zeros(2)
    ylim = np.zeros(2)
    zlim = np.zeros(2)
    system=None
    system_bonds=None
    system_angles=None
    coords=[]
    bonds=[]
    angles=[]

    if type=="atoms" or type=="bonds" or type=="angles":
        at_cols = ['at_id', 'mol_id', 'at_type', 'x', 'y', 'z']
        n_atoms = 0
        at_types = 0
        if type=="bonds" or type=="angles":
            bd_cols = ['bd_id', 'bd_type', 'bd_1', 'bd_2']
            n_bonds = 0
            bd_types = 0
            if type=="angles":
                an_cols = ['an_id', 'an_type', 'an_1', 'an_2', 'an_3']
                n_angles = 0
                an_types = 0

    with (open(file, 'r') as f):
        l = f.readline()
        line=l.split()
        while l:
            if "atoms" in line:
                n_atoms = int(line[0])
            if "atom" in line:
                at_types=int(line[0])
            if "bonds" in line:
                n_bonds = int(line[0])
            if "bond" in line:
                bd_types = line[0]
            if "angles" in line:
                n_angles = int(line[0])
            if "angle" in line:
                an_types = line[0]
            if "xlo" in line:
                xlim = np.array(line)[:2].astype(float)
            if "ylo" in line:
                ylim = np.array(line)[:2].astype(float)
            if "zlo" in line:
                zlim = np.array(line)[:2].astype(float)
            if "Masses" in line:
                line = f.readline()
                for i in range(at_types):
                    line = f.readline().split()
                    masses+=[float(line[1])]
            if "Atoms" in line:
                line = f.readline()
                coords = np.zeros((n_atoms, len(at_cols)))
                for i in range(n_atoms):
                    line = f.readline().split()
                    coords[i] = np.array(line[:len(at_cols)])
            if "Bonds" in line:
                line = f.readline()
                bonds = np.zeros((n_bonds, len(bd_cols)))
                for i in range(n_bonds):
                    line = f.readline().split()
                    bonds[i] = np.array(line)
            if "Angles" in line:
                line = f.readline()
                angles = np.zeros((n_angles, len(an_cols)))
                for i in range(n_angles):
                    line = f.readline().split()
                    angles[i] = np.array(line)
            l = f.readline()
            line = l.split()
    system = pd.DataFrame(coords, columns=at_cols)
    system[['x', 'y', 'z']] = system[['x', 'y', 'z']].astype(float)
    system[['at_id', 'mol_id', 'at_type']] = system[['at_id', 'mol_id', 'at_type']].astype(int)
    system.sort_values('at_id',inplace=True)
    system.reset_index(inplace=True,drop=True)
    if n_bonds>0:
        system_bonds = pd.DataFrame(bonds.astype(int), columns=bd_cols)
        system_bonds.sort_values('bd_id',inplace=True)
        system_bonds.reset_index(inplace=True,drop=True)
    if n_angles>0:
        system_angles = pd.DataFrame(angles.astype(int), columns=an_cols)
        system_angles.sort_values('an_id',inplace=True)
        system_angles.reset_index(inplace=True,drop=True)
    return system, system_bonds, system_angles, masses, xlim, ylim, zlim


def get_details(name):
    els = re.split("[_]", name)  # '[nNfFrcRgrft]', input_dir.name)
    n_beads = int(re.findall(r"[-+]?\d*\.\d+|\d+", els[1])[0])
    functionality = int(re.findall(r"[-+]?\d*\.\d+|\d+", els[2])[0])
    r_core = float(re.findall(r"[-+]?\d*\.\d+|\d+", els[3])[0])
    r_bond = float(re.findall(r"[-+]?\d*\.\d+|\d+", els[4])[0])
    r_cbond = float(re.findall(r"[-+]?\d*\.\d+|\d+", els[5])[0])

    ghost = False
    r_conf = 0.0
    peclet = 0.0
    Dr = 1.0
    Dt = 1.0
    gamma = 1.0
    n_mol = 1
    r_int = 0.0
    brownian = False
    seed_start = None
    for el in els:
        if "mols" in el and r_int==0.0:
            n_mol = int(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "pe" in el:
            peclet = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "rf" in el and r_int==0.0:
            r_conf = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "bro" in el:
            brownian = True
        if "Dt" in el:
            Dt = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "Dr" in el:
            Dr = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "gm" in el:
            gamma = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "gh" in el and r_int==0.0:
            ghost = True
        if "seed" in el:
            seed_start = int(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
        if "r_int" in el:
            r_int = float(re.findall(r"[-+]?\d*\.\d+|\d+", el)[0])
            n_mol=2
            r_conf=0.0
            ghost=False

    details = {"n_beads": n_beads, "n_mol": n_mol, "functionality": functionality,
               "r_core": r_core, "r_bond":r_bond, "r_cbond":r_cbond, "r_conf": r_conf, "peclet": peclet, "brownian": brownian, "Dr": Dr, "Dt": Dt, "gamma": gamma, "r_int":r_int,"seed_start": seed_start,"ghost": ghost}
    # print(details)
    return details


def get_name(details, prefix=True):
    n_beads = details["n_beads"]
    n_mol = details["n_mol"]
    functionality = details["functionality"]
    r_core = details["r_core"]
    peclet = details["peclet"]
    r_conf = details["r_conf"]
    r_bond = details["r_bond"]
    r_cbond = details["r_cbond"]
    Dr = details["Dr"]
    Dt = details["Dt"]
    gamma = details["gamma"]
    r_int = details["r_int"]
    seed_start = details["seed_start"]
    suff = ""
    pref = "mar_"
    if n_mol > 1:
        suff += f"_mols{n_mol:03d}"
        if details["ghost"]:
            suff += "_gh"
    if details["peclet"] > 0:
        suff += f"_pe{peclet:.1f}"
    if details["r_conf"] > 0:
        suff += f"_rf{r_conf:.1f}"
    if details["brownian"] > 0:
        suff += f"_bro"
    if details["Dt"] != 1.0:
        suff += f"_Dt{Dt:.3f}"
    if details["Dr"] != 1.0:
        suff += f"_Dr{Dr:.3f}"
    if details["gamma"] != 1.0:
        suff += f"_gm{gamma:.0f}"
    if details["seed_start"] != None:
        suff += f"_seed{seed_start:d}"
    if details["r_int"] > 0:
        suff += f"_ri{r_int:.1f}"
    if not prefix:
        pref=""
    name = f'{pref}n{n_beads:03d}_f{functionality:03d}_rc{r_core:.2f}_rb{r_bond:.2f}_rcb{r_cbond:.2f}' + suff
    return name