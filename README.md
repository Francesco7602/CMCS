# Comparative Model for Complex Systems (CMCS)

This project is a multi-model epidemiological simulator designed to explore and compare virus dynamics in a population. It provides an interactive web-based interface built with [Solara](https://solara.dev/) for configuring parameters, running simulations, and visualizing results in real-time.

The key feature of this tool is the simultaneous comparison of three distinct modeling paradigms: Agent-Based Modeling (ABM), Ordinary Differential Equations (ODE), and the Gillespie Stochastic Simulation Algorithm (SSA).

## Features

- **Multi-Model Comparison:**
  - **Agent-Based Model (ABM):** Simulates the behavior and interactions of individual agents.
  - **SEIR ODE Model:** A deterministic, continuous model for macroscopic comparison.
  - **Gillespie's Algorithm (SSA):** A discrete-event, stochastic model that is exact for well-mixed populations.
- **Interactive Dashboard:**
  - Adjust simulation parameters on the fly and see results instantly.
  - Live plotting of SEIR curves for all three models.
  - Visualization of agent states on a spatial grid or a network graph.
  - A Petri net representation of the final SEIR state distribution.
- **Advanced ABM Topologies:**
  - **Grid:** Agents are placed on a 2D grid and can move to neighboring cells.
  - **Complex Networks:**
    - **Barabasi-Albert:** A scale-free network.
    - **Watts-Strogatz:** A small-world network.
    - **Erdos-Renyi:** A random graph.
    - **Communities:** A network with distinct communities, enabling targeted interventions.
- **Dynamic & Long-Term Policies:**
  - **Vital Dynamics:** Simulates birth and death processes (`mu` parameter) to model open populations in long-term scenarios (e.g., >1 year).
  - **Vaccination:** Supports multiple strategies:
    - Initial vaccination ('random' or 'targeted' by network centrality).
    - Vaccination of newborns in long-term simulations.
  - **Adaptive Lockdowns:** A dynamic social distancing policy that responds to the percentage of symptomatic infections. It can be applied globally or targeted at specific communities.
- **Analysis Tools:**
  - **Stochastic Analysis:** Run batches of simulations to analyze the distribution of epidemic peaks and calculate risk probabilities.
  - **Parameter Sweep:** Systematically vary a parameter to analyze its impact on the peak number of infections.

## Theoretical Models

The simulator implements an **SEIR** model, which can simulate both closed populations (standard) and open populations with constant turnover. Each agent can be in one of the following states:

- **Susceptible (S):** Can be infected.
- **Exposed (E):** Infected but not yet infectious.
- **Infected (I):** Infectious, subdivided into:
    - **Asymptomatic (I_asymp):** Infectious but shows no symptoms.
    - **Symptomatic (I_symp):** Infectious and shows symptoms (and may self-isolate).
- **Recovered (R):** Immune, either through recovery or vaccination.

### Agent-Based Model (ABM)

The core is an ABM where `VirusAgent` instances interact. This model captures complex, emergent behaviors that are difficult to represent with mathematical equations, such as agent movement, network effects, and adaptive lockdowns. It has been extended to support vital dynamics, where agents can be removed (death) and replaced with new (susceptible or vaccinated) agents over time.

### SEIR ODE Model

For comparison, the simulator solves a deterministic SEIR model using Ordinary Differential Equations. This provides a macroscopic view of the epidemic's average dynamics. The model is extended to include terms for vital dynamics (`mu`) and vaccination of the newborn cohort (`vax_pct`).

### Gillespie's Algorithm (SSA)

This is a stochastic, discrete-event simulation that provides an exact trajectory for the system's state evolution. It is computationally more efficient than an ABM for well-mixed populations. Like the ODE model, it has been extended to handle births, deaths, and newborn vaccination.

## Getting Started

### Prerequisites

- Python 3.7+
- `pip` for package installation

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

The user interface is built with Solara. To run the web application, execute the following command in the project's root directory:

```bash
solara run virus_model.refactored_model.main
```

This will start a web server and open the simulator in your default web browser.

## Code Structure

- `virus_model/refactored_model/main.py`: The main entry point for the Solara web application. It defines the UI and orchestrates the simulations.
- `virus_model/refactored_model/model.py`: Defines the `VirusModel` class for the Agent-Based Model.
- `virus_model/refactored_model/agent.py`: Defines the `VirusAgent` class, representing an individual in the simulation.
- `virus_model/refactored_model/constants.py`: Contains shared constants (e.g., agent states, colors).
- `virus_model/refactored_model/ode.py`: Implements the SEIR ODE model, including vital dynamics.
- `virus_model/refactored_model/gillespie.py`: Implements the Gillespie SSA, including vital dynamics.
- `virus_model/refactored_model/plotting.py`: Contains functions for plotting and saving simulation results.
- `requirements.txt`: A list of Python packages required for the project.
- `README.md`: This file.
