## Analysis Folder

The `analysis/` folder contains scripts used for processing and visualizing the simulation data. These tools allow you to analyze various aspects of the system, such as computing mean squared displacement (MSD), generating videos of particle or polymer dynamics, and more. Below is a description of the key scripts found in this folder:

### Scripts

#### 1. **`msd_calculator_single.py`**
   - **Description**: This script computes and plots the **mean squared displacement (MSD)** for single particle simulations. The MSD is a measure of the particle's diffusion and provides insight into its dynamic behavior.
   - **Functions**:
     - Calculates the MSD over time for active or passive particles.
     - Generates plots of MSD vs. time, which can be used to characterize the diffusive behavior of the particle (e.g., normal diffusion, subdiffusion, or superdiffusion).
    - **Output**: The MSD results and plots will be saved in the `data/03_analyzed/msd/` directory.
   - **Usage**: This script should be used to analyze data from simulations of single particles.
   
#### 2. **`msd_calculator.py`**
   - **Description**: This script calculates and plots the **mean squared displacement (MSD)** for star polymer simulations. The MSD here helps understand how the polymer arms or the entire polymer move over time.
   - **Functions**:
     - Computes MSD for both the center of mass of the star polymer and potentially individual arms.
     - Generates plots that show the diffusion characteristics of the star polymer system.
   - **Output**: The MSD results and plots will be saved in the `data/03_analyzed/msd/` directory.
   - **Usage**: This script is used for analyzing data from simulations of full star polymer systems.

#### 3. **`video_maker_single.py`**
   - **Description**: This script creates movies or animations of the **single particle** dynamics. By generating a visual representation of the particle's movement, you can better understand how it evolves over time, including self-propulsion or confinement effects.
   - **Functions**:
     - Generates video files (e.g., `.mp4`) that show the trajectory of the single particle within the simulation domain.
     - Helps visualize particle motion and the effect of parameters like self-propulsion or confinement.
   - **Output**: The video files will be saved in the `data/03_analyzed/video/` directory.
   - **Usage**: Use this script to visualize the behavior of single particle simulations.

#### 4. **`video_maker.py`**
   - **Description**: This script is used to create movies or animations for **full star polymer** systems. It visualizes the motion and behavior of the entire star polymer, including both the polymer arms and the central core.
   - **Functions**:
     - Generates a video representation of the star polymer simulation, showing how the polymer chains move and interact over time.
     - Useful for visualizing how the star polymer structure changes during the simulation.\
   - **Output**: The video files will be saved in the `data/03_analyzed/video/` directory.
   - **Usage**: Use this script to visualize simulations of star polymer systems.

### Configuration Files

In order to choose which system to analyze (single particles or star polymers), you need to modify the relevant parameters in the corresponding `.json` configuration file:

- **`single_parameters.json`**: This file contains parameters specific to the single particle simulations. Here you can set values like the Peclet number, confinement radius, and dynamics type (Brownian or Langevin).
  
- **`stelle_parameters.json`**: This file contains parameters specific to the star polymer simulations. It includes settings for the number of arms, beads per arm, core radius, and the choice of dynamics (Langevin or Brownian).
