# virus_model/refactored_model/agent.py
#considero pure l'eta?
from mesa import Agent
from mesa.space import MultiGrid, NetworkGrid
from .constants import *

class VirusAgent(Agent):
    """
    Represents an individual agent in the virus simulation.

    Each agent has an age group, a state (susceptible, exposed, etc.), and parameters
    governing their behavior and disease progression.
    """
    def __init__(self, model, age_group, beta, gamma, incubation_mean, prob_symptomatic):
        """
        Initializes a new VirusAgent.

        Args:
            model: The main model instance.
            age_group (int): The age group category of the agent (e.g., CHILD, ADULT).
            beta (float): The transmission rate.
            gamma (float): The recovery rate.
            incubation_mean (int): The average incubation period in days.
            prob_symptomatic (float): The probability of an infected agent developing symptoms.
        """
        super().__init__(model)
        self.age_group = age_group
        self.state = STATE_SUSCEPTIBLE
        self._next_state = STATE_SUSCEPTIBLE
        self.beta = beta
        self.gamma = gamma
        self.mu = MU_GROUP[self.age_group]
        self.prob_symptomatic = prob_symptomatic
        self.incubation_period = max(1, int(self.random.gauss(incubation_mean, 1)))
        self.days_exposed = 0
        self.community = None
        self._neighbors = None

    def step_sensing(self):
        """
        Determines the agent's next state based on its current state and interactions.

        This is the "sensing" part of a simultaneous activation scheduler. The agent
        evaluates its condition and environment to decide on a potential state change,
        but does not apply it yet.
        """
        self._next_state = self.state

        # State transition logic
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
            # Roll for mortality (Case Fatality Rate)
            elif self.random.random() < self.mu:
                self.model.remove_and_respawn(self)

        # Movement logic
        # Symptomatic agents self-isolate and do not move.
        # All others move (if not in lockdown/distancing).
        if self.state != STATE_INFECTED_SYMPTOMATIC:
            if isinstance(self.model.grid, MultiGrid):
                if self.random.random() > self.model.get_social_distancing(self):
                    self.move_candidate()
            elif isinstance(self.model.grid, NetworkGrid):
                if self.random.random() > self.model.get_social_distancing(self):
                    self.move_on_network()

    def step_apply(self):
        """
        Applies the state change determined in the step_sensing phase.

        This method updates the agent's state and the model's compartment counts
        accordingly. It's the "apply" part of a simultaneous activation.
        """
        if self.state != self._next_state:
            # Decrement counter for old state
            if self.state == STATE_SUSCEPTIBLE: self.model.s_count -= 1
            elif self.state == STATE_EXPOSED: self.model.e_count -= 1
            elif self.state == STATE_INFECTED_ASYMPTOMATIC: self.model.i_asymp_count -= 1
            elif self.state == STATE_INFECTED_SYMPTOMATIC: self.model.i_symp_count -= 1
            elif self.state == STATE_RECOVERED: self.model.r_count -= 1
            
            # Increment counter for new state
            if self._next_state == STATE_SUSCEPTIBLE: self.model.s_count += 1
            elif self._next_state == STATE_EXPOSED: self.model.e_count += 1
            elif self._next_state == STATE_INFECTED_ASYMPTOMATIC: self.model.i_asymp_count += 1
            elif self._next_state == STATE_INFECTED_SYMPTOMATIC: self.model.i_symp_count += 1
            elif self._next_state == STATE_RECOVERED: self.model.r_count += 1

        self.state = self._next_state

    def step_sequential(self):
        """
        Executes a full agent step for sequential schedulers.

        This combines the sensing and applying phases into a single call, suitable
        for schedulers that activate agents one by one.
        """
        self.step_sensing()
        self.step_apply()

    def check_exposure(self):
        """
        Checks if a susceptible agent gets exposed to the virus from its neighbors.

        The method iterates through neighbors and, based on the transmission rate (beta),
        determines if an infection event occurs.

        Returns:
            bool: True if the agent becomes exposed, False otherwise.
        """
        if isinstance(self.model.grid, NetworkGrid):
            if self._neighbors is None:
                neighbor_nodes = list(self.model.grid.G.neighbors(self.pos))
                self._neighbors = self.model.grid.get_cell_list_contents(neighbor_nodes)
            neighborhood = self._neighbors
        else:
            # On MultiGrid (dynamic), neighbors must be recalculated each time
            neighborhood = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)
            neighborhood = [a for a in neighborhood if a != self]

        for neighbor in neighborhood:
            if neighbor.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC]:
                if self.random.random() < self.beta:
                    return True
        return False

    def move_candidate(self):
        """
        Moves the agent to a random neighboring cell on the MultiGrid.
        """
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if possible_steps:
            new_pos = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_pos)

    def move_on_network(self):
        """
        Moves the agent to an adjacent node on the NetworkGrid.
        """
        # Get neighboring nodes (excluding the current node)
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            include_center=False,
            radius=1
        )

        # Choose a random node among available neighbors
        if neighborhood:
            new_pos = self.random.choice(neighborhood)
            self.model.grid.move_agent(self, new_pos)