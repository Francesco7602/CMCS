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
  - **Execution Modes:** Choose between "Animated" for step-by-step visualization and "Turbo" for fast, end-of-run results.
  - Live plotting of SEIR curves for all three models.
  - Visualization of agent states on a spatial grid or a network graph.
  - A Transition Schema representation of the final SEIR state distribution.
- **Advanced ABM Topologies:**
  - **Grid:** Agents are placed on a 2D grid and can move to neighboring cells.
  - **Complex Networks:**
    - **Barabasi-Albert:** A scale-free network.
    - **Watts-Strogatz:** A small-world network.
    - **Erdos-Renyi:** A random graph.
    - **Communities:** A network with distinct communities, enabling targeted interventions.
- **Multiple Agent Schedulers:**
  - Includes various agent activation regimes to model different assumptions about time and agent concurrency:
    - **Synchronous (Simultaneous):** All agents sense their environment and then apply their state changes simultaneously.
    - **Uniform (Ordered):** Agents are activated sequentially in a fixed order.
    - **Random (Shuffled):** Agents are activated sequentially in a random order each step.
    - **Poisson (Random Interval):** A random number of agents are activated at each step, determined by a Poisson distribution.
- **Dynamic & Long-Term Policies:**
  - **Vital Dynamics:** Simulates birth and death processes (`mu` parameter) to model open populations in long-term scenarios (e.g., >1 year).
  - **Disease-Specific Mortality:** Models a Case Fatality Rate (`mu_disease`) where infected agents can die from the disease. This mortality rate is age-dependent. Deceased agents are replaced by new susceptible agents to maintain a constant population, distinct from the general birth/death rate.
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

Infected individuals can also die from the disease at a specified rate (`mu_disease`), at which point they are removed from the population and replaced by a new Susceptible individual to maintain population constancy.

### Agent-Based Model (ABM)

The core is an ABM where `VirusAgent` instances interact. This model captures complex, emergent behaviors that are difficult to represent with mathematical equations, such as agent movement, network effects, and adaptive lockdowns. It has been extended to support vital dynamics, where agents can be removed (death) and replaced with new (susceptible or vaccinated) agents over time. It also models a disease-specific fatality rate, where an agent's probability of dying from the infection can depend on its age group.

### SEIR ODE Model

For comparison, the simulator solves a deterministic SEIR model using Ordinary Differential Equations. This provides a macroscopic view of the epidemic's average dynamics. The model is extended to include terms for vital dynamics (`mu`) and vaccination of the newborn cohort (`vax_pct`). It further includes a term for disease-specific mortality (`mu_disease`), where deaths from infection are balanced by new susceptibles to maintain a constant population.

### Gillespie's Algorithm (SSA)

This is a stochastic, discrete-event simulation that provides an exact trajectory for the system's state evolution. It is computationally more efficient than an ABM for well-mixed populations. Like the ODE model, it has been extended to handle births, deaths, and newborn vaccination. It also models disease-specific death as a distinct event type, where an infected individual is replaced by a susceptible one.

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
