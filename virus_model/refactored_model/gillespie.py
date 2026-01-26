# virus_model/refactored_model/gillespie.py

import numpy as np
import pandas as pd


def run_gillespie_simulation(N, beta, gamma, sigma, max_steps, mu=0.0, vax_pct=0.0):
    """
    Executes a SEIR simulation using the Gillespie Stochastic Simulation Algorithm (SSA)
    extended with Vital Dynamics (Births/Deaths).

    Events handling constant population N:
    - Infection, Progression, Recovery (Standard SEIR)
    - Vital Dynamics: An agent dies and is immediately replaced (Respawn).
      The replacement is vaccinated with probability 'vax_pct', else Susceptible.
    """
    # Stato iniziale
    # Assumiamo che la vaccinazione iniziale sia gestita fuori (nel conteggio S, R iniziali)
    # Tuttavia, per coerenza con l'ABM, qui partiamo da una condizione pulita se non passata diversamente.
    # Per semplicità, replichiamo lo stato "standard" con 1 esposto.
    # NOTA: Se volessi passare lo stato iniziale esatto, dovremmo aggiungere argomenti.
    # Qui approssimiamo partendo da 1 E e il resto S/R in base a vax_pct se mu > 0 (o standard se mu=0).

    # Calcolo S, R iniziali approssimati
    initial_vaccinated = int(N * vax_pct) if vax_pct > 0 else 0
    E = 1
    I = 0
    S = N - 1 - initial_vaccinated
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

        # Rate Vitali (Respawn logic)
        # Ogni compartimento ha un tasso di morte mu * X.
        # Quando uno muore, rinasce come S (prob 1-p) o R (prob p).
        # Consideriamo solo le transizioni che CAMBIANO lo stato.
        # Es: S muore e rinasce S -> Nessun cambio (ignoriamo).
        # Es: S muore e rinasce R -> S-1, R+1 (Rate: mu * S * p)

        p = vax_pct
        q = 1.0 - p

        # S muore -> diventa R
        rate_S_to_R = mu * S * p

        # E muore -> diventa S oppure R
        rate_E_to_S = mu * E * q
        rate_E_to_R = mu * E * p

        # I muore -> diventa S oppure R
        rate_I_to_S = mu * I * q
        rate_I_to_R = mu * I * p

        # R muore -> diventa S
        rate_R_to_S = mu * R * q

        # Somma totale propensities
        total_rate = (rate_infection + rate_progression + rate_recovery +
                      rate_S_to_R + rate_E_to_S + rate_E_to_R +
                      rate_I_to_S + rate_I_to_R + rate_R_to_S)

        if total_rate == 0:
            # Nessun evento possibile (es. tutto estinto o parametri a 0)
            break

        # --- 2. Tempo al prossimo evento (tau) ---
        r1 = np.random.rand()
        tau = (1.0 / total_rate) * np.log(1.0 / r1)

        # Se il salto temporale supera il tempo massimo, ci fermiamo
        if t + tau > max_steps:
            break

        t += tau

        # --- 3. Scelta dell'evento ---
        r2 = np.random.rand()
        threshold = r2 * total_rate

        current_sum = rate_infection
        if threshold < current_sum:
            # Infection: S -> E
            S -= 1;
            E += 1
        else:
            current_sum += rate_progression
            if threshold < current_sum:
                # Progression: E -> I
                E -= 1;
                I += 1
            else:
                current_sum += rate_recovery
                if threshold < current_sum:
                    # Recovery: I -> R
                    I -= 1;
                    R += 1
                else:
                    # --- Eventi Vitali ---
                    current_sum += rate_S_to_R
                    if threshold < current_sum:
                        S -= 1;
                        R += 1
                    else:
                        current_sum += rate_E_to_S
                        if threshold < current_sum:
                            E -= 1;
                            S += 1
                        else:
                            current_sum += rate_E_to_R
                            if threshold < current_sum:
                                E -= 1;
                                R += 1
                            else:
                                current_sum += rate_I_to_S
                                if threshold < current_sum:
                                    I -= 1;
                                    S += 1
                                else:
                                    current_sum += rate_I_to_R
                                    if threshold < current_sum:
                                        I -= 1;
                                        R += 1
                                    else:
                                        current_sum += rate_R_to_S
                                        if threshold < current_sum:
                                            R -= 1;
                                            S += 1

        # Aggiorna history
        t_hist.append(t)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)

    # --- FIX GRAFICO: Padding finale ---
    # Se il loop è finito prima di max_steps (es. estinzione o total_rate=0),
    # aggiungiamo un punto finale per estendere la linea piatta nel grafico.
    if t < max_steps:
        t_hist.append(max_steps)
        S_hist.append(S)
        E_hist.append(E)
        I_hist.append(I)
        R_hist.append(R)

    return pd.DataFrame({'time': t_hist, 'S': S_hist, 'E': E_hist, 'I': I_hist, 'R': R_hist})