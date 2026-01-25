# virus_model/refactored_model/gillespie.py

import numpy as np
import pandas as pd

def run_gillespie_simulation(N, beta, gamma, sigma, max_steps):
    """
        Executes a SEIR simulation using the Gillespie Stochastic Simulation Algorithm (SSA).

        KEY DIFFERENCES FROM ODE:
        1. Discrete Variables: S, E, I, R are integers, not continuous floats.
        2. Stochastic Time: The time step 'tau' is not fixed; it is generated randomly
           based on the probability of an event occurring.
        3. Randomness: Each run produces a unique trajectory.

        Args:
        N (int): Popolazione totale.
        beta (float): Tasso di contagio.
        gamma (float): Tasso di guarigione.
        sigma (float): Tasso di progressione (1 / periodo di incubazione).
        max_steps (int): Numero massimo di step temporali della simulazione ABM/ODE,
                         usato come tempo massimo (t_max).

        Returns:
            pandas.DataFrame: DataFrame con le colonne ['time', 'S', 'E', 'I', 'R'].
    """
    # Stato iniziale
    S = N - 1
    E = 1
    I = 0
    R = 0
    t = 0.0

    # History
    history = {'time': [t], 'S': [S], 'E': [E], 'I': [I], 'R': [R]}
    # Loop until max time/steps or until the epidemic dies out (no Exposed or Infected left)
    while t < max_steps and (E > 0 or I > 0):
        # Calculate Propensities (a_mu)
        # The propensity a_mu * dt is the probability that reaction mu
        # occurs in the next infinitesimal time interval dt
        prop_infection = (beta * S * I) / N
        prop_progression = sigma * E
        prop_recovery = gamma * I
        # a_0: Total propensity (sum of all reaction rates)
        total_propensity = prop_infection + prop_progression + prop_recovery

        if total_propensity == 0:
            break

        # Determine time to next event (tau)
        # The time until the next reaction is exponentially distributed
        # with parameter a_0
        # Formula: tau = (1/a_0) * ln(1/r1) where r1 is uniform (0,1)
        r1 = np.random.rand()
        tau = (1 / total_propensity) * np.log(1 / r1)

        # Select WHICH reaction occurs
        # Reaction mu is chosen with probability a_mu / a_0
        # We use a Monte Carlo selection method
        r2 = np.random.rand()
        event_threshold = r2 * total_propensity

        if event_threshold < prop_infection:
            # Evento: Infezione (S -> E)
            S -= 1
            E += 1
        elif event_threshold < prop_infection + prop_progression:
            # Evento: Progressione (E -> I)
            E -= 1
            I += 1
        else:
            # Evento: Guarigione (I -> R)
            I -= 1
            R += 1

        # Aggiorna il tempo e salva lo stato
        t += tau
        history['time'].append(t)
        history['S'].append(S)
        history['E'].append(E)
        history['I'].append(I)
        history['R'].append(R)

    return pd.DataFrame(history)
