# virus_model/refactored_model/ode.py
#nel pacchetto di slide 3 la formula e diversda...
#puoi estender il modello con una versione births and death
#e con la versione che considera le vqaccinazioni (anche se forse gia lo faccio in qualche modo)
"""
-------------------------------------------------------------------------
NOTE: Difference between SIR and SEIR Models
-------------------------------------------------------------------------

The fundamental difference lies in the inclusion of the 'Exposed' (E)
compartment, which represents the incubation/latency period.

1. SIR Model (S -> I -> R):
   - Assumes instantaneous infectiousness.
   - Once a Susceptible individual contacts the virus, they immediately
     become Infected (I) and capable of spreading it.
   - Governed only by Beta (infection rate) and Gamma (recovery rate).

2. SEIR Model (S -> E -> I -> R) [CURRENTLY IMPLEMENTED]:
   - Introduces a latency period.
   - Upon contact, a Susceptible individual becomes 'Exposed' (E).
   - In the 'E' state, the individual is infected but NOT yet infectious
     (they cannot spread the virus to others).
   - Transition from E -> I occurs at rate Sigma (σ),
     where σ = 1 / mean_incubation_period.

Impact on Simulation:
The SEIR model results in a delayed epidemic peak compared to the SIR model
because the 'Exposed' state acts as a buffer, slowing down the initial
spread of the infection.
-------------------------------------------------------------------------
"""
def seir_ode(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt
