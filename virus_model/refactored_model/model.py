# virus_model/refactored_model/model.py

import networkx as nx
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
                 ws_k=4, ws_p=0.1, er_p=0.1, comm_l=5, comm_k=20,
                 vaccine_strategy="none", vaccine_pct=0.0,
                 scheduler_type="random",
                 prob_symptomatic=0.6):

        super().__init__()
        
        self.topology = topology
        if topology == "communities":
            self.N = comm_l * comm_k
        else:
            self.N = N

        self.width = width
        self.height = height
        self.scheduler_type = scheduler_type
        self.steps_count = 0
        self.incubation_mean = incubation_mean
        self.social_distancing = 0.0
        self.lockdown_active = False
        self.prob_symptomatic = prob_symptomatic

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
                self.G = nx.barabasi_albert_graph(self.N, 5)
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

        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: sum(1 for a in m.agents if a.state == STATE_SUSCEPTIBLE),
                "E": lambda m: sum(1 for a in m.agents if a.state == STATE_EXPOSED),
                "I_asymp": lambda m: sum(1 for a in m.agents if a.state == STATE_INFECTED_ASYMPTOMATIC),
                "I_symp": lambda m: sum(1 for a in m.agents if a.state == STATE_INFECTED_SYMPTOMATIC),
                "R": lambda m: sum(1 for a in m.agents if a.state == STATE_RECOVERED),
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
                self.community_social_distancing[i] = min(0.9, (pct_detected / 0.2) * 0.9)
                if self.community_social_distancing[i] > 0.1:
                    self.lockdown_active = True
        else:
            infected_detected = sum(1 for a in self.agents if a.state == STATE_INFECTED_SYMPTOMATIC)
            pct_detected = infected_detected / self.N

            # Adaptive lockdown
            self.social_distancing = min(0.9, (pct_detected / 0.2) * 0.9)
            if self.social_distancing > 0.1:
                self.lockdown_active = True
            else:
                self.lockdown_active = False

        self.datacollector.collect(self)

        if self.scheduler_type == "random":
            self.agents.shuffle_do("step_sequential")
        else:
            self.agents.do("step_sensing")
            self.agents.do("step_apply")
        self.steps_count += 1

    def get_social_distancing(self, agent):
        if self.topology == "communities" and agent.community is not None:
            return self.community_social_distancing[agent.community]
        return self.social_distancing
