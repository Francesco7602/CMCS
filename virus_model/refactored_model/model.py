# virus_model/refactored_model/model.py
"""
This file defines the main Agent-Based Model (ABM), `VirusModel`.

The `VirusModel` class encapsulates the entire simulation world, including the agents,
the spatial environment (grid or network), and the rules governing their interactions
and state transitions. It manages the simulation step, data collection, and the
implementation of interventions like vaccination and lockdowns.
"""

import networkx as nx
import numpy as np
from mesa import Model
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector
from networkx.algorithms import community

from .agent import VirusAgent
from .constants import *


class VirusModel(Model):
    """
    The main model for the virus spread simulation.

    It initializes the environment, creates agents, and manages the simulation loop.
    """
    def __init__(self, N=300, width=20, height=20,
                 beta=0.5, gamma=0.1, incubation_mean=3,
                 topology="grid",
                 ws_k=4, ws_p=0.1, er_p=0.1, ba_m=2, comm_l=5, comm_k=20,
                 vaccine_strategy="none", vaccine_pct=0.0,
                 scheduler_type="random",
                 prob_symptomatic=0.6,
                 lockdown_threshold_pct=0.2, lockdown_max_sd=0.9, lockdown_active_threshold=0.1, mu=0.0):
        """
        Initializes the VirusModel.

        Args:
            N (int): The number of agents.
            width (int), height (int): Dimensions for the grid topology.
            beta (float): Transmission rate.
            gamma (float): Recovery rate.
            incubation_mean (int): Average incubation period.
            topology (str): The spatial structure ('grid', 'watts_strogatz', etc.).
            ws_k (int), ws_p (float): Parameters for Watts-Strogatz graphs.
            er_p (float): Parameter for Erdős-Rényi graphs.
            ba_m (int): Parameter for Barabási-Albert graphs.
            comm_l (int), comm_k (int): Parameters for community graphs.
            vaccine_strategy (str): Vaccination strategy ('none', 'random', 'targeted').
            vaccine_pct (float): Percentage of the population to vaccinate.
            scheduler_type (str): The agent activation regime ('random', 'simultaneous', etc.).
            prob_symptomatic (float): Probability of developing symptoms.
            lockdown_threshold_pct (float): Infection percentage to trigger lockdown.
            lockdown_max_sd (float): Maximum social distancing factor during lockdown.
            lockdown_active_threshold (float): Threshold to deactivate lockdown.
            mu (float): Natural birth/death rate for vital dynamics.
        """
        super().__init__()
        self.layout = None
        self.mu = mu
        self.vaccine_pct = vaccine_pct
        self.np_random = np.random.default_rng(self.random.randint(0, 2**32 - 1))
        
        self.topology = topology
        if topology == "communities":
            self.N = comm_l * comm_k
        else:
            self.N = N
        self.total_agents_created = N
        self.width = width
        self.height = height
        self.scheduler_type = scheduler_type
        self.steps_count = 0
        self.beta = beta
        self.gamma = gamma
        self.incubation_mean = incubation_mean
        self.social_distancing = 0.0
        self.lockdown_active = False
        self.prob_symptomatic = prob_symptomatic
        self.lockdown_threshold_pct = lockdown_threshold_pct
        self.lockdown_max_sd = lockdown_max_sd
        self.lockdown_active_threshold = lockdown_active_threshold

        # --- Compartment Counts ---
        self.s_count = 0
        self.e_count = 0
        self.i_asymp_count = 0
        self.i_symp_count = 0
        self.r_count = 0

        self.communities = None
        self.community_social_distancing = {}
        self.total_deaths = 0

        # --- Environment Setup ---
        if topology in ["network", "watts_strogatz", "erdos_renyi", "communities"]:
            if topology == "watts_strogatz":
                self.G = nx.watts_strogatz_graph(self.N, k=ws_k, p=ws_p)
            elif topology == "erdos_renyi":
                self.G = nx.erdos_renyi_graph(self.N, p=er_p)
            elif topology == "communities":
                self.G = nx.connected_caveman_graph(l=comm_l, k=comm_k)
                self.communities = list(community.greedy_modularity_communities(self.G))
                for i, comm in enumerate(self.communities):
                    self.community_social_distancing[i] = 0.0
            else:  # Default to Barabasi-Albert
                self.G = nx.barabasi_albert_graph(self.N, ba_m)
            self.grid = NetworkGrid(self.G)
        else: # Default to grid
            self.G = None
            self.grid = MultiGrid(width, height, torus=True)

        if self.G is not None:
            # Pre-calculate layout for network visualizations to improve performance.
            self.layout = nx.spring_layout(self.G, seed=42)

        # --- Agent Creation ---
        for i in range(self.N):
            age_group = self.random.choices(
                [CHILD, TEEN, ADULT, SENIOR],
                weights=[0.1, 0.1, 0.6, 0.2]
            )[0]

            a = VirusAgent(self, age_group, beta, gamma, incubation_mean, prob_symptomatic)
            if self.communities:
                for comm_id, comm in enumerate(self.communities):
                    if i in comm:
                        a.community = comm_id
                        break
            self.agents.add(a)
            if self.G is not None:
                self.grid.place_agent(a, i)
            else:
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(a, (x, y))

        # --- Initial Conditions ---
        self.apply_vaccination(vaccine_strategy, vaccine_pct)

        susceptible_agents = [a for a in self.agents if a.state == STATE_SUSCEPTIBLE]
        if susceptible_agents:
            patient_zero = self.random.choice(susceptible_agents)
            patient_zero.state = STATE_EXPOSED

        # --- Initialize Compartment Counts ---
        for a in self.agents:
            if a.state == STATE_SUSCEPTIBLE: self.s_count += 1
            elif a.state == STATE_EXPOSED: self.e_count += 1
            elif a.state == STATE_RECOVERED: self.r_count += 1
            
        # --- Data Collection ---
        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: m.s_count,
                "E": lambda m: m.e_count,
                "I_asymp": lambda m: m.i_asymp_count,
                "I_symp": lambda m: m.i_symp_count,
                "R": lambda m: m.r_count,
                "Deaths": lambda m: m.total_deaths,
                "TotalAgents": lambda m: m.total_agents_created,
                "Lockdown": lambda m: 1 if m.lockdown_active else 0
            }
        )

    def apply_vaccination(self, strategy, pct):
        """
        Applies a vaccination strategy to a percentage of the population.

        Args:
            strategy (str): The vaccination strategy ('random' or 'targeted').
            pct (float): The fraction of the population to vaccinate.
        """
        num_vax = int(self.N * pct)
        if num_vax == 0: return
        
        targets = []
        if strategy == "random":
            targets = self.random.sample(list(self.agents), num_vax)
        elif strategy == "targeted":
            if self.G is not None: # Targeted vaccination based on node degree in networks
                degrees = dict(self.G.degree())
                sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
                top_nodes = sorted_nodes[:num_vax]
                targets = self.grid.get_cell_list_contents(top_nodes)
            else: # Fallback to random for grid topology
                targets = self.random.sample(list(self.agents), num_vax)
                
        for a in targets:
            if a.state == STATE_SUSCEPTIBLE:
                a.state = STATE_RECOVERED

    def step(self):
        """
        Executes one time step of the simulation.
        
        This involves updating lockdown status, collecting data, activating agents
        according to the chosen scheduler, and handling vital dynamics (births/deaths).
        """
        # --- Lockdown and Social Distancing Logic ---
        if self.topology == "communities":
            self.lockdown_active = False
            for i, comm in enumerate(self.communities):
                comm_agents = self.grid.get_cell_list_contents(list(comm))
                if not comm_agents: continue
                infected_detected = sum(1 for a in comm_agents if a.state == STATE_INFECTED_SYMPTOMATIC)
                pct_detected = infected_detected / len(comm_agents)
                
                # Hysteresis mechanism for lockdown
                if pct_detected >= self.lockdown_threshold_pct:
                    self.community_social_distancing[i] = self.lockdown_max_sd
                elif pct_detected < (self.lockdown_threshold_pct * 0.5):
                    self.community_social_distancing[i] = 0.0

                if self.community_social_distancing[i] > 0:
                    self.lockdown_active = True
        else: # Global lockdown logic
            pct_detected = self.i_symp_count / self.N
            
            if pct_detected >= self.lockdown_threshold_pct:
                self.social_distancing = self.lockdown_max_sd
                self.lockdown_active = True
            elif pct_detected < (self.lockdown_threshold_pct * 0.5): # Hysteresis
                self.social_distancing = 0.0
                self.lockdown_active = False

        self.datacollector.collect(self)

        # --- Agent Activation ---
        if self.scheduler_type == "random":
            self.agents.shuffle_do("step_sequential")
        elif self.scheduler_type == "simultaneous":
            self.agents.do("step_sensing")
            self.agents.do("step_apply")
        elif self.scheduler_type == "uniform":
            self.agents.do("step_sequential")
        elif self.scheduler_type == "poisson":
            active_fraction = np.random.poisson(0.5) / self.N
            n_active = max(1, int(active_fraction * self.N))
            active_agents = self.random.sample(list(self.agents), min(n_active, len(self.agents)))
            for agent in active_agents:
                agent.step_sequential()
        else: # Fallback to simultaneous
            self.agents.do("step_sensing")
            self.agents.do("step_apply")

        # --- Vital Dynamics (Births/Deaths) ---
        if self.mu > 0:
            n_deaths = self.np_random.binomial(self.N, self.mu)
            if n_deaths > 0:
                agents_list = list(self.agents)
                dying_agents = self.random.sample(agents_list, n_deaths)

                for agent in dying_agents:
                    self.remove_and_respawn(agent)
        
        self.steps_count += 1

    def get_social_distancing(self, agent):
        """
        Returns the social distancing factor for a given agent.
        
        This can be global or community-specific depending on the topology.

        Args:
            agent (VirusAgent): The agent for which to get the factor.

        Returns:
            float: The social distancing factor (0.0 to 1.0).
        """
        if self.topology == "communities" and agent.community is not None:
            return self.community_social_distancing.get(agent.community, 0.0)
        return self.social_distancing

    def remove_and_respawn(self, agent, reason="natural"):
        """
        Handles the death and immediate replacement of an agent to maintain population N.

        This can be triggered by disease-specific mortality or natural causes. The new
        "born" agent is reset and can be either susceptible or vaccinated.

        Args:
            agent (VirusAgent): The agent to be removed and respawned.
        """
        self.total_agents_created += 1
        if reason == "disease":
            self.total_deaths += 1
        # 1. Decrement count for the agent's old state
        if agent.state == STATE_SUSCEPTIBLE: self.s_count -= 1
        elif agent.state == STATE_EXPOSED: self.e_count -= 1
        elif agent.state == STATE_INFECTED_ASYMPTOMATIC: self.i_asymp_count -= 1
        elif agent.state == STATE_INFECTED_SYMPTOMATIC: self.i_symp_count -= 1
        elif agent.state == STATE_RECOVERED: self.r_count -= 1

        # 2. Respawn: Reset the agent's state (can be born vaccinated)
        if self.random.random() < self.vaccine_pct:
            agent.state = STATE_RECOVERED
            agent._next_state = STATE_RECOVERED
            self.r_count += 1
        else:
            agent.state = STATE_SUSCEPTIBLE
            agent._next_state = STATE_SUSCEPTIBLE
            self.s_count += 1
        
        # Reset internal agent counters
        agent.days_exposed = 0
