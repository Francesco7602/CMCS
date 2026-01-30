# virus_model/refactored_model/gillespie.py

import numpy as np
import pandas as pd


def run_gillespie_simulation(N, beta, gamma, sigma, max_steps, mu=0.0, mu_disease=0.0, vax_pct=0.0):
    """
    Executes a SEIR simulation using the Gillespie Stochastic Simulation Algorithm (SSA)
    extended with Vital Dynamics (Births/Deaths).

    Events handling constant population N:
    - Infection, Progression, Recovery (Standard SEIR)
    - Disease-specific death (I -> S)
    - Vital Dynamics: An agent dies and is immediately replaced (Respawn).
      The replacement is vaccinated with probability 'vax_pct', else Susceptible.
    """
    # Stato iniziale
    initial_infected_count = 5
    initial_vaccinated = int(N * vax_pct) if vax_pct > 0 else 0
    E = initial_infected_count
    I = 0
    S = N - E - I - initial_vaccinated
    R = initial_vaccinated

    t = 0.0

    # History lists
    t_hist = [t]
    S_hist = [S]
    E_hist = [E]
    I_hist = [I]
    R_hist = [R]

    while t < max_steps:
        # --- 1. Calcolo Propensities (Rates) ---
        # Rate Epidemici
        rate_infection = (beta * S * I) / N
        rate_progression = sigma * E
        rate_recovery = gamma * I
        rate_disease_death = mu_disease * I  # Morte per malattia

        # Rate Vitali (Respawn logic)
        p = vax_pct
        q = 1.0 - p
        rate_S_to_R = mu * S * p
        rate_E_to_S = mu * E * q
        rate_E_to_R = mu * E * p
        rate_I_to_S = mu * I * q
        rate_I_to_R = mu * I * p
        rate_R_to_S = mu * R * q

        # Somma totale propensities
        total_rate = (rate_infection + rate_progression + rate_recovery + rate_disease_death +
                      rate_S_to_R + rate_E_to_S + rate_E_to_R +
                      rate_I_to_S + rate_I_to_R + rate_R_to_S)

        if total_rate == 0:
            break

        # --- 2. Tempo al prossimo evento (tau) ---
        r1 = np.random.rand()
        tau = (1.0 / total_rate) * np.log(1.0 / r1)

        if t + tau > max_steps:
            break
        t += tau

        # --- 3. Scelta dell'evento ---
        r2 = np.random.rand()
        threshold = r2 * total_rate

        current_sum = rate_infection
        if threshold < current_sum:
            S -= 1; E += 1  # Infection
        else:
            current_sum += rate_progression
            if threshold < current_sum:
                E -= 1; I += 1  # Progression
            else:
                current_sum += rate_recovery
                if threshold < current_sum:
                    I -= 1; R += 1  # Recovery
                else:
                    current_sum += rate_disease_death
                    if threshold < current_sum:
                        I -= 1; S += 1  # Disease Death & Respawn
                    else:
                        # --- Eventi Vitali ---
                        current_sum += rate_S_to_R
                        if threshold < current_sum: S -= 1; R += 1
                        else:
                            current_sum += rate_E_to_S
                            if threshold < current_sum: E -= 1; S += 1
                            else:
                                current_sum += rate_E_to_R
                                if threshold < current_sum: E -= 1; R += 1
                                else:
                                    current_sum += rate_I_to_S
                                    if threshold < current_sum: I -= 1; S += 1
                                    else:
                                        current_sum += rate_I_to_R
                                        if threshold < current_sum: I -= 1; R += 1
                                        else:
                                            current_sum += rate_R_to_S
                                            if threshold < current_sum: R -= 1; S += 1

        # Aggiorna history
        t_hist.append(t)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)

    if t < max_steps:
        t_hist.append(max_steps)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)

    return pd.DataFrame({'time': t_hist, 'S': S_hist, 'E': E_hist, 'I': I_hist, 'R': R_hist})