# Computational LAMMPS Project: 2D Star Polymers Simulation

This project simulates the behavior of star polymers in 2D using the LAMMPS molecular dynamics software. Star polymers are a class of branched macromolecules where several polymer chains (arms) are attached to a central core. The star polymer model in this project is flexible and allows for various modifications such as changing the number of arms, the number of beads per arm, and the core radius. Additionally, it is possible to choose between Langevin and Brownian dynamics.

## How to Create and Run the Simulation

1. **Compile LAMMPS:**
    - download the last version of LAMMPS from https://www.lammps.org/
    - in lammps-your-version/src/ activate the following packages with make yes-'package': RIGID, MOLECULE, BROWNIAN, GRANULAR, DIPOLE
    - copy the files in src/lammps_custom to lammps-your-version/src/
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
   - Specifies the number of star polymer molecules to simulate.
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

### 4. **Molecule-Molecule Interaction Flag (`gh`)**
   - Controls whether the molecules interact with each other.
   - **Example:** `gh = 1` means the molecules will not interact.
   - **Range:** `0` or `1`.

### 5. **Radius of Confinement (`r_conf`)**
   - Specifies the radius of confinement for the polymer system. If set to 0, there is no confinement.
   - **Example:** `r_conf = 0.0` means there will be no confinement.
   - **Range:** Any non-negative real number.

### 6. **Peclet Number (`peclet`)**
   - Defines the Peclet number, which controls the strength of self-propulsion for active particles. A higher Peclet number results in stronger propulsion.
   - **Example:** `peclet = 10.0` means the self-propulsion strength is set to 10.
   - **Range:** Any positive real number.

### 7. **Bond Size (`r_bond`)**
   - Specifies the size of the bond between adjacent beads in the polymer chain.
   - **Example:** `r_bond = 17/15` means the bead-bead bond size is 17/15.
   - **Range:** Any positive real number.

### 8. **Arm Grafting Distance (`r_cbond`)**
   - Defines the distance between the arm grafting location and the center of the core.
   - **Example:** `r_cbond = (10 - 1.5) / 15` means the arm grafting distance is calculated as `(10 - 1.5) / 15`.
   - **Range:** Any positive real number.

### 9. **Dynamics Type (`brownian`)**
   - Determines the type of dynamics to be used in the simulation:
     - `0`: Langevin dynamics
     - `1`: Brownian dynamics
   - **Example:** `brownian = 1` means Brownian dynamics will be used.
   - **Range:** `0` or `1`.

### 10. **Timestep (`dt`)**
   - Specifies the timestep used in the simulation.
   - **Example:** `dt = 0.001` means the simulation will use a timestep of 0.001.
   - **Range:** Any positive real number.
   - **Note**: More overdamped systems require a smaller dt

### 11. **Adjustment for Brownian Dynamics**
   - If `brownian = 1`, the timestep is automatically adjusted to a smaller value (0.0001) for numerical stability in Brownian dynamics simulations.
   - **Example:** If `brownian = 1`, `dt` will be automatically set to `0.0001`.
   - **Range:** This adjustment applies only when `brownian = 1`.

### 12. **Traslational Diffusion Coefficient (`Dt`)**

   - Specifies the **traslational diffusion coefficient**. This parameter governs how quickly a particle undergoes random traslational motion due to thermal fluctuations.
   - **Example**: `Dt = 0.1` sets the rotational diffusion coefficient to 0.1 (in simulation-specific units), producing Brownian dynamics.
   - **Range**: Any positive real number.

### 13. **Rotational Diffusion Coefficient (`Dr`)**

   - Specifies the **rotational diffusion coefficient**. This parameter governs how quickly a particle’s orientation undergoes random rotational motion due to thermal fluctuations.
   - **Example**: `Dr = 0.8` sets the rotational diffusion coefficient to 0.8 (in simulation-specific units), producing moderate rotational randomization during Brownian dynamics.
   - **Range**: Any positive real number.

### 14. **Friction Coefficient (`gamma`)**

   - Specifies the **friction coefficient**. This parameter is important in Langevin simulations, as it determins if the system is over- or underdamped.
   - **Example**: `gamma = 100` sets the friction coefficient to 100 (in simulation-specific units), producing overdamped dynamics.
   - **Range**: Any positive real number.

### 15. **Number of Arms (`fs`)**

   - Specifies the **functionality** of the star polymer, i.e., how many arms extend from the central bead. This parameter (and with little modification also the others) can also be provided as a **vector**, allowing multiple simulations to be run with different functionalities in a single batch.
   - **Example:**
     - `fs = [6]` sets the polymer to have **6 arms**.
     - `fs = [4, 6, 8]` would initialize three simulations, one for each listed functionality.
   - **Range:** Any positive integer (or a list of positive integers).
   - **Note:** When a vector is provided, the simulation framework will iterate over each value, enabling systematic exploration of how star functionality affects polymer properties.


### 16. **Interaction Distance (`r_int`)**

   - Allows to study **effective interactions** between stars at distance `r_int`, expressed in terms of the **sigma** parameter.
   - **Example**: `r_int = 150/sigma` sets the star cores apart of a distance `r_int` and fixes their positions.
   - **Range**: Any positive real number.
   - **Note**: If `r_int > 0`, this automatically sets the following parameters:
     - `mol = 2`: only 2 stars allowed.
     - `gh = 0`: interactions between stars must be active.
     - `rconf = 0`: no confinement allowed.
     - This will also output the forces active on the core.

### 17. **Contact Friction (`contact`)**

   - If set to 1, introduces **contact friction** between particles and between the particles and the wall.
   - **Example**: `contact = 1` turns on contact friction.
   - **Range**: `0` or `1`.

