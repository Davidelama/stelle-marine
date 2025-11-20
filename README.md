# Computational LAMMPS Project: 2D Star Polymers Simulation

This project simulates the behavior of star polymers in 2D using the LAMMPS molecular dynamics software. The simulation framework provides a flexible model for studying the dynamics of star-shaped macromolecules, which are characterized by several polymer arms branching from a central core. This setup can be modified to study a wide range of star polymer configurations and dynamic behaviors under various conditions.

The simulation framework also allows users to simulate single particles, which can help test the system in simple conditions

## Star Polymers "stelle"

Star polymers are a class of branched macromolecules that consist of multiple polymer chains, also known as arms, radiating from a central core. These macromolecules are commonly studied due to their unique structure and potential applications in materials science, drug delivery, and self-assembly. 

In this project, the star polymer model is flexible and allows various modifications:
- **Number of arms**: The number of polymer chains attached to the central core can be varied.
- **Number of beads per arm**: Each arm is composed of a series of beads, and the number of beads in each arm can be adjusted to simulate different molecular weights.
- **Core radius**: The radius of the central core can be customized to explore its effect on the overall structure of the polymer.
- **Dynamics**: The simulation can switch between **Langevin dynamics** (dissipative particle dynamics) and **Brownian dynamics**, which control how the polymers interact with the surrounding environment and each other.

Star polymers are studied here in a **2D confinement** to simplify the computational requirements and focus on the core behavior of these systems, including their structure, mobility, and response to external fields such as confinement or active propulsion.

## Single Particles "single"

In addition to simulating star polymers, the project also allows the simulation of single particles, which are simpler but still interesting systems for studying basic dynamic behaviors, particularly in confinement or under active propulsion.

### Use Cases for Single Particles:
- **Active Particles**: Single particles can be modeled as active Brownian particles, where each particle experiences a self-propulsion force governed by a Peclet number. This is particularly relevant in studies of active matter and motility-driven phase transitions.
- **Confinement**: Single particles can also be confined in a specific radius, allowing studies of particle behavior under spatial constraints.
- **Dynamics**: The simulation can switch between **Langevin dynamics** (dissipative particle dynamics) and **Brownian dynamics**, which control how the polymers interact with the surrounding environment and each other.

## Folder Structure

The project is organized into several directories, each serving a specific purpose. Below is a breakdown of the folder structure:

├── bin/
├── data/
│ ├── 01_raw/
│ ├── 02_processed/
│ └── 03_analyzed/
└── src/
  ├── stelle/
  ├── single/
  └── analysis/

### 1. **`bin/`**
   - This folder contains executable programs and binary files required to run the simulations.
   - **LAMMPS** and any other necessary compiled executables for running the simulations will be found here.
   - **Example:** `lammps` binary, other simulation-related executables.

### 2. **`data/`**
   This directory is used for storing simulation data at various stages of processing:

   - **`01_raw/`**: 
     - This folder contains the raw data from the simulations.
     - When simulations are run, output files and intermediate data files are saved here.
     - **Example**: Configuration files, trajectory data, raw outputs from LAMMPS simulations.

   - **`02_processed/`**:
     - This folder contains data that has been processed and cleaned for further analysis.
     - Post-processing scripts are used to filter or format the raw data into a more useful form for analysis.
     - **Example**: Processed trajectories, cleaned data files, or transformed coordinates.

   - **`03_analyzed/`**:
     - This folder stores the final results of the analysis, such as visualizations and statistical data.
     - This is where you would find plots, graphs, or any other final outputs after the data has been analyzed.
     - **Example**: Final plots, graphs, statistical summaries, and figures for publications or presentations.

### 3. **`src/`**
   The `src` directory contains the source code for the simulation scripts and analysis tools. It is subdivided into three main folders:

   - **`stelle/`**:
     - This folder contains the scripts for simulating **star polymers**.
     - These scripts handle the initialization, setup, and running of simulations specifically for star polymer systems.
     - **Example**: Scripts for defining the star polymer model, setting parameters, and running LAMMPS simulations.
   
   - **`single/`**:
     - This folder contains the scripts for simulating **single bead** systems.
     - These scripts focus on running simulations of single particles (e.g., active particles, Brownian particles) under various conditions such as confinement and self-propulsion.
     - **Example**: Scripts for initializing single particle dynamics, running simulations, and outputting results.

   - **`analysis/`**:
     - This folder contains the scripts used for **data analysis** and **visualization**.
     - These scripts process the raw simulation data (found in the `data/02_processed/` folder) and generate plots, graphs, or perform statistical analysis.
     - **Example**: Scripts for plotting radial distributions, time correlation functions, or generating movies of the polymer behavior.

---

### Summary of Folder Contents:
- **`bin/`**: Executables and binary files (e.g., LAMMPS binaries).
- **`data/`**:
  - `01_raw/`: Raw simulation output.
  - `02_processed/`: Processed simulation data.
  - `03_analyzed/`: Final analysis results (plots, figures).
- **`src/`**:
  - `stelle/`: Scripts for star polymer simulations.
  - `single/`: Scripts for single particle simulations.
  - `analysis/`: Scripts for data analysis and visualization.

This structure is designed to keep the project organized and allow for easy access to the different stages of simulation and analysis.

