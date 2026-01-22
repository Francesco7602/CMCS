# virus_model/refactored_model/gillespie.py

import numpy as np
import pandas as pd

def run_gillespie_simulation(N, beta, gamma, sigma, max_steps):
    """
    Esegue una simulazione SEIR usando l'algoritmo di Gillespie (SSA).

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

    while t < max_steps and (E > 0 or I > 0):
        # 1. Calcolo delle propensities (tassi di reazione)
        prop_infection = (beta * S * I) / N
        prop_progression = sigma * E
        prop_recovery = gamma * I

        total_propensity = prop_infection + prop_progression + prop_recovery

        if total_propensity == 0:
            break

        # 2. Calcolo del tempo al prossimo evento (tau)
        r1 = np.random.rand()
        tau = (1 / total_propensity) * np.log(1 / r1)

        # 3. Selezione del prossimo evento
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
