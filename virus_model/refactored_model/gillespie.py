# virus_model/refactored_model/gillespie.py

import numpy as np
import pandas as pd


def run_gillespie_simulation(N, beta, gamma, sigma, max_steps, mu=0.0, mu_disease=0.0, vax_pct=0.0, lockdown_enabled=False, lockdown_thresh=0.2, p_lock=1.0):
    """
    Executes a SEIR simulation using the Gillespie Stochastic Simulation Algorithm (SSA).

    This version is extended to include vital dynamics (births and deaths),
    disease-specific mortality, vaccination, and dynamic lockdowns. The total
    population N is kept constant.

    Args:
        N (int): Total population size.
        beta (float): The transmission rate.
        gamma (float): The recovery rate.
        sigma (float): The rate of progression from exposed to infected.
        max_steps (int): The maximum simulation time.
        mu (float, optional): The natural death rate (and birth rate). Defaults to 0.0.
        mu_disease (float, optional): The disease-specific death rate for infected individuals. Defaults to 0.0.
        vax_pct (float, optional): The fraction of newborns who are vaccinated. Defaults to 0.0.
        lockdown_enabled (bool, optional): If True, dynamic lockdown rules are applied. Defaults to False.
        lockdown_thresh (float, optional): The infected population fraction to trigger a lockdown. Defaults to 0.2.
        p_lock (float, optional): The reduction factor for beta during a lockdown (e.g., 1.0 means no reduction). Defaults to 1.0.

    Returns:
        pd.DataFrame: A DataFrame containing the time series of S, E, I, and R compartments.
    """
    # Initial state
    initial_infected_count = 5
    initial_vaccinated = int(N * vax_pct) if vax_pct > 0 else 0
    E = initial_infected_count
    I = 0
    S = N - E - I - initial_vaccinated
    R = N - S - E - I

    t = 0.0
    D = 0
    cumulative_births = 0


    # History lists to store simulation trajectory
    t_hist = [t]
    S_hist = [S]
    E_hist = [E]
    I_hist = [I]
    R_hist = [R]
    D_hist = [D]

    while t < max_steps:
        # Dynamic lockdown threshold check for Gillespie
        p_t = p_lock if (lockdown_enabled and (I / N) >= lockdown_thresh) else 1.0

        # --- 1. Calculate Propensities (Rates) for all possible events ---
        # Recalculate infection rate with current p_t
        rate_infection = (beta * p_t * S * I) / N
        
        # Epidemic rates
        rate_progression = sigma * E
        rate_recovery = gamma * I
        rate_disease_death = mu_disease * I  # Death due to disease

        # Vital Dynamics rates (Respawn logic)
        p = vax_pct
        q = 1.0 - p
        rate_S_to_R = mu * S * p  # Susceptible dies, replaced by Recovered (vaccinated)
        rate_E_to_S = mu * E * q  # Exposed dies, replaced by Susceptible
        rate_E_to_R = mu * E * p  # Exposed dies, replaced by Recovered
        rate_I_to_S = mu * I * q  # Infected dies, replaced by Susceptible
        rate_I_to_R = mu * I * p  # Infected dies, replaced by Recovered
        rate_R_to_S = mu * R * q  # Recovered dies, replaced by Susceptible

        # Sum of all propensities
        total_rate = (rate_infection + rate_progression + rate_recovery + rate_disease_death +
                      rate_S_to_R + rate_E_to_S + rate_E_to_R +
                      rate_I_to_S + rate_I_to_R + rate_R_to_S)

        if total_rate == 0:
            break

        # --- 2. Determine time to next event (tau) ---
        r1 = np.random.rand()
        tau = (1.0 / total_rate) * np.log(1.0 / r1)

        if t + tau > max_steps:
            break
        t += tau

        # --- 3. Choose which event occurs ---
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
                    # DA QUI IN GIÙ SONO TUTTE MORTI CON RIMPIAZZO (QUINDI NASCITE)
                    cumulative_births += 1
                    current_sum += rate_disease_death
                    if threshold < current_sum:
                        I -= 1; S += 1; D += 1  # Disease Death & Respawn as Susceptible
                    else:
                        # --- Vital Dynamics Events ---
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
                                            # This must be rate_R_to_S
                                            R -= 1; S += 1

        # Update history
        t_hist.append(t)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)
        D_hist.append(D)

    # Ensure the history extends to max_steps for consistent plotting
    if t < max_steps:
        t_hist.append(max_steps)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)
        D_hist.append(D)

    return pd.DataFrame({'time': t_hist, 'S': S_hist, 'E': E_hist, 'I': I_hist, 'R': R_hist, 'Deaths': D_hist}), cumulative_births