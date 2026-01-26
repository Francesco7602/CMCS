# virus_model/refactored_model/ode.py

def seir_ode(y, t, N, beta, sigma, gamma, mu=0.0, vax_pct=0.0):
    S, E, I, R = y

    # Flussi vitali (Nati)
    births = mu * N
    new_vaccinated = births * vax_pct          # Quota p dei neonati (slide 66)
    new_susceptibles = births * (1 - vax_pct)  # I restanti sono suscettibili

    # Flussi Epidemici
    infection = beta * S * I / N
    progression = sigma * E
    recovery = gamma * I

    # Morti naturali (proporzionali alla popolazione attuale del comparto)
    d_S_death = mu * S
    d_E_death = mu * E
    d_I_death = mu * I
    d_R_death = mu * R

    # Equazioni Differenziali
    dSdt = new_susceptibles - infection - d_S_death
    dEdt = infection - progression - d_E_death
    dIdt = progression - recovery - d_I_death
    dRdt = recovery + new_vaccinated - d_R_death

    return dSdt, dEdt, dIdt, dRdt
