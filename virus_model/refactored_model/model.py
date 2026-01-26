# virus_model/refactored_model/model.py

import networkx as nx
import numpy as np
from mesa import Model
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector

from networkx.algorithms import community

from .agent import VirusAgent
from .constants import (
    STATE_SUSCEPTIBLE,
    STATE_EXPOSED,
    STATE_INFECTED_ASYMPTOMATIC,
    STATE_INFECTED_SYMPTOMATIC,
    STATE_RECOVERED,
)


class VirusModel(Model):
    def __init__(self, N=300, width=20, height=20,
                 beta=0.5, gamma=0.1, incubation_mean=3,
                 topology="grid",
                 ws_k=4, ws_p=0.1, er_p=0.1, ba_m=2, comm_l=5, comm_k=20,
                 vaccine_strategy="none", vaccine_pct=0.0,
                 scheduler_type="random",
                 prob_symptomatic=0.6,
                 lockdown_threshold_pct=0.2, lockdown_max_sd=0.9, lockdown_active_threshold=0.1, mu=0.0):

        super().__init__()
        self.mu = mu
        self.vaccine_pct = vaccine_pct
        self.np_random = np.random.default_rng(self.random.randint(0, 2**32 - 1))
        
        self.topology = topology
        if topology == "communities":
            self.N = comm_l * comm_k
        else:
            self.N = N

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

        self.s_count = 0
        self.e_count = 0
        self.i_asymp_count = 0
        self.i_symp_count = 0
        self.r_count = 0

        self.communities = None
        self.community_social_distancing = {}

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
        else:
            self.G = None
            self.grid = MultiGrid(width, height, torus=True)

        for i in range(self.N):
            a = VirusAgent(self, beta, gamma, incubation_mean, prob_symptomatic)
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

        self.apply_vaccination(vaccine_strategy, vaccine_pct)

        susceptible_agents = [a for a in self.agents if a.state == STATE_SUSCEPTIBLE]
        if susceptible_agents:
            patient_zero = self.random.choice(susceptible_agents)
            patient_zero.state = STATE_EXPOSED

        for a in self.agents:
            if a.state == STATE_SUSCEPTIBLE: self.s_count += 1
            elif a.state == STATE_EXPOSED: self.e_count += 1
            elif a.state == STATE_RECOVERED: self.r_count += 1
            
        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: m.s_count,
                "E": lambda m: m.e_count,
                "I_asymp": lambda m: m.i_asymp_count,
                "I_symp": lambda m: m.i_symp_count,
                "R": lambda m: m.r_count,
                "Lockdown": lambda m: 1 if m.lockdown_active else 0
            }
        )

    def apply_vaccination(self, strategy, pct):
        num_vax = int(self.N * pct)
        if num_vax == 0: return
        targets = []
        if strategy == "random":
            targets = self.random.sample(list(self.agents), num_vax)
        elif strategy == "targeted" and self.G is not None:
            degrees = dict(self.G.degree())
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            top_nodes = sorted_nodes[:num_vax]
            targets = self.grid.get_cell_list_contents(top_nodes)
        elif strategy == "targeted":
            targets = self.random.sample(list(self.agents), num_vax)
        for a in targets:
            a.state = STATE_RECOVERED

    def step(self):
        if self.topology == "communities":
            self.lockdown_active = False
            for i, comm in enumerate(self.communities):
                comm_agents = self.grid.get_cell_list_contents(list(comm))
                infected_detected = sum(1 for a in comm_agents if a.state == STATE_INFECTED_SYMPTOMATIC)
                pct_detected = infected_detected / len(comm_agents)
                self.community_social_distancing[i] = min(self.lockdown_max_sd, (pct_detected / self.lockdown_threshold_pct) * self.lockdown_max_sd)
                if self.community_social_distancing[i] > self.lockdown_active_threshold:
                    self.lockdown_active = True
        else:
            infected_detected = self.i_symp_count
            pct_detected = infected_detected / self.N

            # Adaptive lockdown
            self.social_distancing = min(self.lockdown_max_sd, (pct_detected / self.lockdown_threshold_pct) * self.lockdown_max_sd)
            if self.social_distancing > self.lockdown_active_threshold:
                self.lockdown_active = True
            else:
                self.lockdown_active = False

        self.datacollector.collect(self)

        if self.scheduler_type == "random":
            self.agents.shuffle_do("step_sequential")
        else:
            self.agents.do("step_sensing")
            self.agents.do("step_apply")

        # --- NUOVA LOGICA: VITAL DYNAMICS (Nascite/Morti) ---
        if self.mu > 0:
            # Calcola quanti muoiono in questo step (distribuzione binomiale)
            n_deaths = self.np_random.binomial(self.N, self.mu)

            if n_deaths > 0:
                # Scegli a caso chi muore
                agents_list = list(self.agents)
                dying_agents = self.random.sample(agents_list, n_deaths)

                for agent in dying_agents:
                    # 1. Aggiorna contatori (rimuovi stato vecchio)
                    if agent.state == 0: self.s_count -= 1
                    elif agent.state == 1: self.e_count -= 1
                    elif agent.state == 2: self.i_asymp_count -= 1
                    elif agent.state == 3: self.i_symp_count -= 1
                    elif agent.state == 4: self.r_count -= 1

                    # 2. Respawn (Il "figlio" prende il posto del "morto")
                    # Decide se nasce vaccinato
                    if self.random.random() < self.vaccine_pct:
                        agent.state = 4 # STATE_RECOVERED
                        agent._next_state = 4
                        self.r_count += 1
                    else:
                        agent.state = 0 # STATE_SUSCEPTIBLE
                        agent._next_state = 0
                        self.s_count += 1

                    # Reset contatori interni dell'agente
                    agent.days_exposed = 0
        
        self.steps_count += 1

    def get_social_distancing(self, agent):
        if self.topology == "communities" and agent.community is not None:
            return self.community_social_distancing[agent.community]
        return self.social_distancing
