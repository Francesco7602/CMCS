# virus_model/refactored_model/agent.py

from mesa import Agent
from mesa.space import MultiGrid, NetworkGrid
from .constants import (
    STATE_SUSCEPTIBLE,
    STATE_EXPOSED,
    STATE_INFECTED_ASYMPTOMATIC,
    STATE_INFECTED_SYMPTOMATIC,
    STATE_RECOVERED,
)

class VirusAgent(Agent):
    def __init__(self, model, beta, gamma, incubation_mean, prob_symptomatic):
        super().__init__(model)
        self.state = STATE_SUSCEPTIBLE
        self._next_state = STATE_SUSCEPTIBLE
        self.beta = beta
        self.gamma = gamma
        self.prob_symptomatic = prob_symptomatic
        self.incubation_period = max(1, int(self.random.gauss(incubation_mean, 1)))
        self.days_exposed = 0
        self.community = None

    def step_sensing(self):
        self._next_state = self.state

        # Logica di transizione di stato
        if self.state == STATE_SUSCEPTIBLE:
            if self.check_exposure():
                self._next_state = STATE_EXPOSED
        
        elif self.state == STATE_EXPOSED:
            self.days_exposed += 1
            if self.days_exposed >= self.incubation_period:
                if self.random.random() < self.prob_symptomatic:
                    self._next_state = STATE_INFECTED_SYMPTOMATIC
                else:
                    self._next_state = STATE_INFECTED_ASYMPTOMATIC
        
        elif self.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC]:
            if self.random.random() < self.gamma:
                self._next_state = STATE_RECOVERED

        # Logica di movimento
        # Gli agenti sintomatici si auto-isolano e non si muovono.
        # Tutti gli altri si muovono (se non in lockdown/distanziamento).
        if self.state != STATE_INFECTED_SYMPTOMATIC:
            if isinstance(self.model.grid, MultiGrid):
                if self.random.random() > self.model.get_social_distancing(self):
                    self.move_candidate()

    def step_apply(self):
        self.state = self._next_state

    def step_sequential(self):
        self.step_sensing()
        self.step_apply()

    def check_exposure(self):
        if isinstance(self.model.grid, NetworkGrid):
            neighbor_nodes = list(self.model.grid.G.neighbors(self.pos))
            neighbors = self.model.grid.get_cell_list_contents(neighbor_nodes)
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        for neighbor in neighbors:
            if neighbor.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC]:
                if self.random.random() < self.beta:
                    return True
        return False

    def move_candidate(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if possible_steps:
            new_pos = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_pos)
