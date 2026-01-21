# virus_model/refactored_model/model.py

import networkx as nx
from mesa import Model
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector

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
                 vaccine_strategy="none", vaccine_pct=0.0,
                 scheduler_type="random",
                 prob_symptomatic=0.6):

        super().__init__()
        self.N = N
        self.width = width
        self.height = height
        self.topology = topology
        self.scheduler_type = scheduler_type
        self.steps_count = 0
        self.incubation_mean = incubation_mean
        self.social_distancing = 0.0
        self.lockdown_active = False
        self.prob_symptomatic = prob_symptomatic

        if topology == "network":
            self.G = nx.barabasi_albert_graph(N, 5)
            self.grid = NetworkGrid(self.G)
        else:
            self.G = None
            self.grid = MultiGrid(width, height, torus=True)

        for i in range(N):
            a = VirusAgent(self, beta, gamma, incubation_mean, prob_symptomatic)
            self.agents.add(a)
            if topology == "network":
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
        elif strategy == "targeted" and self.topology == "network":
            degrees = dict(self.G.degree())
            sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
            top_nodes = sorted_nodes[:num_vax]
            targets = self.grid.get_cell_list_contents(top_nodes)
        elif strategy == "targeted":
            targets = self.random.sample(list(self.agents), num_vax)
        for a in targets:
            a.state = STATE_RECOVERED

    def step(self):
        infected_detected = sum(1 for a in self.agents if a.state == STATE_INFECTED_SYMPTOMATIC)
        pct_detected = infected_detected / self.N

        if not self.lockdown_active and pct_detected > 0.10:
            self.lockdown_active = True
            self.social_distancing = 0.8
        elif self.lockdown_active and pct_detected < 0.02:
            self.lockdown_active = False
            self.social_distancing = 0.0

        self.datacollector.collect(self)

        if self.scheduler_type == "random":
            self.agents.shuffle_do("step_sequential")
        else:
            self.agents.do("step_sensing")
            self.agents.do("step_apply")
        self.steps_count += 1
