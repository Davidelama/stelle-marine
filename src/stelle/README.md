# Computational LAMMPS Project: 2D Star Polymers Simulation

This project simulates the behavior of star polymers in 2D using the LAMMPS molecular dynamics software. Star polymers are a class of branched macromolecules where several polymer chains (arms) are attached to a central core. The star polymer model in this project is flexible and allows for various modifications such as changing the number of arms, the number of beads per arm, and the core radius. Additionally, it is possible to choose between Langevin and Brownian dynamics.

## How to Create and Run the Simulation

1. **Compile LAMMPS:**
    - download the last version of LAMMPS from https://www.lammps.org/
    - in lammps-you-version/src/ activate the following packages with make yes-'package': ASPHERE, RIGID, MOLECULE, BROWNIAN
    - compile with make mpi and copy lmp_mpi as lmp_mpi_25 in the bin/ folder

1. **Python Packages:**
    - in order to create the initial configurations and run the analysis you will need python (e.g. 3.12) and the following packages:
    - scipy, pandas, matplotlib, jinja2, celluloid

2. **Build the Polymer System:**
    - In src/stelle edit the jobber.py script by initializing parameters as described in the "parameters" section. Then run the script with
      ```bash
      python jobber.py
      ```

3. **Run the Simulation:**
    - After setting up the system, run the LAMMPS simulation using the input file as for example shown in execute_sims.sh or submit a .pbs/.slrm job.

## Parameters Overview

### 1. **Number of Molecules (`n_mol`)**
   - Specifies the number of star polymer molecules to simulate. (Not yet fully implemented)
   - **Example:** `n_mol = 1` means 1 molecule will be generated in the system.
   - **Range:** Any positive integer.

### 2. **Number of Beads per Arm (`n_beads`)**
   - Defines the number of beads in each polymer arm.
   - **Example:** `n_beads = 13` means each polymer arm will consist of 13 beads.
   - **Range:** Any positive integer.

### 3. **Core Radius (`r_core`)**
   - Specifies the radius of the central core in units of bead diameters.
   - **Example:** `r_core = 2/3` means the core will have a radius of 2/3 bead diameters.
   - **Range:** Any positive real number.

### 4. **Number of Restarts (`n_restarts`)**
   - Sets the number of times the simulation will automatically restart to avoid long-term numerical instabilities (especially in large cluster environments).
   - **Example:** `n_restarts = 0` means the simulation will not automatically restart.
   - **Range:** Any non-negative integer.

### 5. **Molecule-Molecule Interaction Flag (`gh`)**
   - Controls whether the molecules interact with each other (currently not fully implemented).
   - **Example:** `gh = 1` means the molecules will not interact.
   - **Range:** `0` or `1`.

### 6. **Radius of Confinement (`r_conf`)**
   - Specifies the radius of confinement for the polymer system. If set to 0, there is no confinement.
   - **Example:** `r_conf = 0.0` means there will be no confinement.
   - **Range:** Any non-negative real number.

### 7. **Peclet Number (`peclet`)**
   - Defines the Peclet number, which controls the strength of self-propulsion for active particles. A higher Peclet number results in stronger propulsion.
   - **Example:** `peclet = 10.0` means the self-propulsion strength is set to 10.
   - **Range:** Any positive real number.

### 8. **Bond Size (`r_bond`)**
   - Specifies the size of the bond between adjacent beads in the polymer chain.
   - **Example:** `r_bond = 17/15` means the bead-bead bond size is 17/15.
   - **Range:** Any positive real number.

### 9. **Arm Grafting Distance (`r_cbond`)**
   - Defines the distance between the arm grafting location and the center of the core.
   - **Example:** `r_cbond = (10 - 1.5) / 15` means the arm grafting distance is calculated as `(10 - 1.5) / 15`.
   - **Range:** Any positive real number.

### 10. **Dynamics Type (`brownian`)**
   - Determines the type of dynamics to be used in the simulation:
     - `0`: Langevin dynamics
     - `1`: Brownian dynamics
   - **Example:** `brownian = 1` means Brownian dynamics will be used.
   - **Range:** `0` or `1`.

### 11. **Timestep (`dt`)**
   - Specifies the timestep used in the simulation.
   - **Example:** `dt = 0.001` means the simulation will use a timestep of 0.001.
   - **Range:** Any positive real number.

### 12. **Adjustment for Brownian Dynamics**
   - If `brownian = 1`, the timestep is automatically adjusted to a smaller value (0.0001) for numerical stability in Brownian dynamics simulations.
   - **Example:** If `brownian = 1`, `dt` will be automatically set to `0.0001`.
   - **Range:** This adjustment applies only when `brownian = 1`.

