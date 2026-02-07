# virus_model/refactored_model/ode.py+
import numpy as np
"""
This file defines the system of Ordinary Differential Equations (ODEs) for a
SEIR (Susceptible-Exposed-Infected-Recovered) model.

The ODEs provide a deterministic, aggregate view of the epidemic dynamics, serving
as a baseline for comparison with the stochastic, individual-based models (ABM
and Gillespie).
"""

def seir_ode(y, t, N, beta, sigma, gamma, mu=0.0, mu_disease=0.0, vax_pct=0.0, lockdown_enabled=False, lockdown_thresh=0.2, p_lock=1.0):
    """
    Defines the differential equations for the SEIR model with vital dynamics and interventions.

    This function is designed to be used with an ODE solver like `scipy.integrate.odeint`.

    Args:
        y (tuple): A tuple `(S, E, I, R)` representing the current population in each compartment.
        t (float): The current time (required by the ODE solver, but not used directly here).
        N (int): The total population size.
        beta (float): The transmission rate.
        sigma (float): The rate of progression from exposed to infected (1 / incubation period).
        gamma (float): The recovery rate.
        mu (float, optional): The natural birth and death rate. Defaults to 0.0.
        mu_disease (float, optional): The disease-specific death rate. Defaults to 0.0.
        vax_pct (float, optional): The fraction of newborns who are vaccinated. Defaults to 0.0.
        lockdown_enabled (bool, optional): Whether dynamic lockdowns are active. Defaults to False.
        lockdown_thresh (float, optional): The infection threshold to trigger a lockdown. Defaults to 0.2.
        p_lock (float, optional): The reduction factor for beta during a lockdown. Defaults to 1.0.

    Returns:
        tuple: A tuple `(dSdt, dEdt, dIdt, dRdt)` representing the rate of change for each compartment.
    """
    S, E, I, R = y

    # Dynamic p(t): Smooth transition using a Sigmoid function
    p_t = 1.0
    if lockdown_enabled:
        # 'k' controlla la rapidità della transizione (più alto = più ripido, ma non istantaneo)
        k = 100
        infection_ratio = I / N

        # Funzione Sigmoide: va da 0 a 1 in modo continuo attorno alla soglia
        activation = 1 / (1 + np.exp(-k * (infection_ratio - lockdown_thresh)))

        # Interpola tra 1.0 (normale) e p_lock (lockdown)
        p_t = 1.0 - (1.0 - p_lock) * activation

    # Vital dynamics (Births)
    births = mu * N
    new_vaccinated = births * vax_pct          # A fraction 'p' of newborns are vaccinated
    new_susceptibles = births * (1 - vax_pct)  # The rest are susceptible

    # Epidemic flows
    infection = (beta * p_t) * S * I / N
    progression = sigma * E
    recovery = gamma * I
    disease_death = mu_disease * I  # Specific death rate due to the disease

    # Natural deaths (proportional to the current population of the compartment)
    d_S_death = mu * S
    d_E_death = mu * E
    d_I_death = mu * I
    d_R_death = mu * R

    # --- Differential Equations ---
    
    # Individuals who die from the disease are replaced by new susceptibles
    dSdt = new_susceptibles - infection - d_S_death + disease_death
    dEdt = infection - progression - d_E_death
    dIdt = progression - recovery - d_I_death - disease_death
    dRdt = recovery + new_vaccinated - d_R_death

    return dSdt, dEdt, dIdt, dRdt
