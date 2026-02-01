# virus_model/refactored_model/ode.py

def seir_ode(y, t, N, beta, sigma, gamma, mu=0.0, mu_disease=0.0, vax_pct=0.0, lockdown_enabled=False, lockdown_thresh=0.2, p_lock=1.0):
    S, E, I, R = y

    # Calcolo p(t) dinamico: se gli infetti superano la soglia, applica p_lock
    # Usiamo la percentuale (I / N) per confrontarla con la soglia della UI
    p_t = 1.0
    if lockdown_enabled and (I / N) >= lockdown_thresh:
        p_t = p_lock

    infection = (beta * p_t) * S * I / N

    # Flussi vitali (Nati)
    births = mu * N
    new_vaccinated = births * vax_pct          # Quota p dei neonati (slide 66)
    new_susceptibles = births * (1 - vax_pct)  # I restanti sono suscettibili

    # Flussi Epidemici
    #infection = beta * S * I / N
    progression = sigma * E
    recovery = gamma * I
    disease_death = mu_disease * I  # Morte specifica per malattia

    # Morti naturali (proporzionali alla popolazione attuale del comparto)
    d_S_death = mu * S
    d_E_death = mu * E
    d_I_death = mu * I
    d_R_death = mu * R

    # Equazioni Differenziali
    # I morti per malattia vengono rimpiazzati da nuovi suscettibili
    dSdt = new_susceptibles - infection - d_S_death + disease_death
    dEdt = infection - progression - d_E_death
    dIdt = progression - recovery - d_I_death - disease_death
    dRdt = recovery + new_vaccinated - d_R_death

    return dSdt, dEdt, dIdt, dRdt
