## Single Particle Simulations

The single particle simulation framework models individual particles in a 2D system. These simulations can be used to study the behavior of active particles, including their motion, self-propulsion, and interactions with confinement. The following parameters are used to configure the single particle simulation:

## How to Create and Run the Simulation

1. **See the README in src/stelle:**

### Parameters

### 1. **Radius of Confinement (`r_conf`)**
   - **Description**: The radius of the confinement in which the particle is constrained. If `r_conf` is set to `0`, the particle is free to move in the entire simulation domain with no confinement.
   - **Example**: `r_conf = 0` means the particle is free and not confined.
   - **Range**: Any non-negative real number. If set to `0`, there is no confinement.
   - **Note**: Confinement is useful for simulating particles in bounded environments, such as confined colloidal particles or active particles in a container.

### 2. **Peclet Number (`peclet`)**
   - **Description**: The Peclet number controls the strength of the self-propulsion force acting on the particle. It is a dimensionless number that characterizes the ratio of the particle's self-propulsion to the diffusive motion.
   - **Example**: `peclet = 10.0` means the self-propulsion strength is set to 10, implying strong self-driven motion.
   - **Range**: Any positive real number.
   - **Note**: A higher Peclet number leads to more pronounced active motion, whereas a low Peclet number makes the particle's behavior more diffusive (similar to passive Brownian motion).

### 3. **Dynamics Type (`brownian`)**
   - **Description**: Determines the type of dynamics used for the simulation:
     - `0`: Langevin dynamics, where the particle interacts with a solvent and experiences both frictional damping and random thermal noise.
     - `1`: Brownian dynamics, where the particle undergoes random motion due to thermal fluctuations, but with no explicit solvent interactions.
   - **Example**: `brownian = 1` selects Brownian dynamics, which is appropriate for modeling the motion of individual particles under thermal fluctuations.
   - **Range**: 
     - `0` for Langevin dynamics.
     - `1` for Brownian dynamics.
   - **Note**: The choice of dynamics affects the way the particle's motion is simulated and should be selected based on the physical model being applied.

### 4. **Rotational Diffusion Coefficient (`Dr`)**

   - **Description**: Specifies the **rotational diffusion coefficient** used *only when* `brownian = 1`. This parameter governs how quickly a particle’s orientation undergoes random rotational motion due to thermal fluctuations.
   - **Example**: `Dr = 0.8` sets the rotational diffusion coefficient to 0.8 (in simulation-specific units), producing moderate rotational randomization during Brownian dynamics.
   - **Range**: Any positive real number.
   - **Note**: This parameter is **ignored** when `brownian = 0` (Langevin dynamics).

---