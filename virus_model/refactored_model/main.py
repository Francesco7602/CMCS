# virus_model/refactored_model/main.py
"""The slides provide a hierarchy that explains this perfectly:

    Petri Nets are a Modelling Language (a way to write down the rules).

    Transition Systems are a Model of Behavior (the resulting graph of all possible states).

Transition System (The "State Graph")

A Transition System is a graph where each node represents the entire state of the world at a specific moment.

    In your model: A single state in a Transition System would be the tuple: s=(S=490,E=5,I=5,R=0).

    The Arrow: A transition arrow would connect that state to a completely different state node, e.g., s′=(S=489,E=6,I=5,R=0).

    The Problem: Because you have N=500 agents, the number of possible states is massive (combinatorial explosion). Drawing a Transition System for your virus model would result in a graph with millions of nodes that is impossible to read.

Petri Net (The "Rules Diagram")

A Petri Net is a compact way to describe how parts of the state interact.

    Places (Circles): These represent local states or "buckets" (Susceptible, Exposed, Infected, Recovered).

    Tokens (Numbers/Dots): These represent the agents. The "State" of the system is defined by how many tokens are in each place.

    Transitions (Rectangles): These represent the events (Infection β, Progression σ, Recovery γ) that move tokens from one place to another."""
import solara
import solara.lab
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import asyncio
import networkx as nx
from scipy.integrate import odeint

from .constants import (
    STATE_EMPTY, SHORT_LABELS, AGENT_COLORS, GRID_CMAP, OUTPUT_DIR,
    STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC,
)
from .ode import seir_ode
from .model import VirusModel
from .plotting import save_single_run_results, save_batch_results_plot, draw_petri_net, save_sweep_results_plot
from .gillespie import run_gillespie_simulation

LONG_TERM_THRESHOLD = 365

# Tutti i parametri del modello ora sono qui
sim_params = solara.reactive({
    "N": 500,
    "steps": 100,
    "grid_size": 30,
    # Epidemiologia
    "beta": 0.6,
    "gamma": 0.1,
    "prob_symp": 0.6,
    "incubation_mean": 3,
    "mu": 0.0,
    # Topologia
    "topology": "grid",
    "ws_k": 4, "ws_p": 0.1,
    "er_p": 0.1,
    "ba_m": 2,
    "comm_l": 5, "comm_k": 20,
    # Interventi
    "vax_strat": "none",
    "vax_pct": 0.0,
    "lockdown_enabled": False,  # Nuovo toggle UI
    "lockdown_thresh": 0.2,  # % popolazione infetta per attivare
    "lockdown_max_sd": 0.8,  # Distanziamento massimo
    # Sistema
    "scheduler": "simultaneous",
    "speed_mode": "animata",  # Opzioni: "animata", "turbo"
})

stochastic_params = solara.reactive({
    "runs": 50,
    "threshold": 150,
})

sweep_params = solara.reactive({
    "parameter": "beta",
    "start": 0.1,
    "end": 1.0,
    "num_steps": 10,
})

# Stati Risultati
model_instance = solara.reactive(None)
run_data = solara.reactive(None)
ode_data = solara.reactive(None)
gillespie_data = solara.reactive(None)
stochastic_results = solara.reactive(None)
sweep_results = solara.reactive(None)

# Stati Interfaccia
is_running = solara.reactive(False)
is_analyzing = solara.reactive(False)
is_sweeping = solara.reactive(False)
status_msg = solara.reactive("")


# --- LOGICA SIMULAZIONE ---

async def run_live_simulation():
    """Esegue una simulazione live (Tab 1)."""
    if is_running.value: return
    is_running.set(True)
    status_msg.set("")

    p = sim_params.value
    steps = p["steps"]
    sigma = 1.0 / p["incubation_mean"]

    # LOGICA SOGLIA TEMPORALE
    is_long_term = steps > LONG_TERM_THRESHOLD

    # CORREZIONE: Se è lungo termine ma mu è 0, forza un valore di default (es. 80 anni)
    if is_long_term and p["mu"] == 0.0:
        current_mu = 1.0 / (80 * 365.0)  # Default 80 anni
    else:
        current_mu = p["mu"] if is_long_term else 0.0

    # 1. Inizializza Modello ABM (passando il mu dinamico)
    model = VirusModel(
        N=p["N"], width=p["grid_size"], height=p["grid_size"],
        beta=p["beta"], gamma=p["gamma"], incubation_mean=p["incubation_mean"],
        topology=p["topology"],
        ws_k=p["ws_k"], ws_p=p["ws_p"], er_p=p["er_p"], ba_m=p["ba_m"],
        comm_l=p["comm_l"], comm_k=p["comm_k"],
        vaccine_strategy=p["vax_strat"], vaccine_pct=p["vax_pct"],
        scheduler_type=p["scheduler"], prob_symptomatic=p["prob_symp"],
        # Parametri Lockdown (passiamo valori fittizi se disabilitato)
        lockdown_threshold_pct=p["lockdown_thresh"] if p["lockdown_enabled"] else 1.0,
        lockdown_max_sd=p["lockdown_max_sd"] if p["lockdown_enabled"] else 0.0,
        lockdown_active_threshold=0.05,
        mu=current_mu  # <--- Passa il mu calcolato
    )
    model_instance.set(model)

    # 2. ODE & Gillespie (Background)
    CORRECT_N = model.N
    t_ode = np.linspace(0, steps, steps)

    # Calcolo Condizioni Iniziali (Vaccinazione Statica al tempo 0)
    # Questa c'è SEMPRE se vax_strat != 'none', sia breve che lungo termine.
    initial_vaccinated = int(CORRECT_N * p["vax_pct"]) if p["vax_strat"] != "none" else 0
    y0 = (CORRECT_N - 1 - initial_vaccinated, 1, 0, initial_vaccinated)

    # Parametro vaccinazione ODE per i NUOVI NATI
    # Se siamo nel breve termine (mu=0), vax_pct_ode deve essere 0 (non nascono bambini).
    # Se siamo nel lungo termine (mu>0), vax_pct_ode è la % di vaccinazione.
    vax_pct_ode = p["vax_pct"] if (is_long_term and p["vax_strat"] != "none") else 0.0

    # Chiamata ODE aggiornata
    ret = odeint(seir_ode, y0, t_ode, args=(CORRECT_N, p["beta"], sigma, p["gamma"], current_mu, vax_pct_ode))
    #ret = odeint(seir_ode, y0, t_ode, args=(CORRECT_N, p["beta"], sigma, p["gamma"]))
    ode_curr = {"t": t_ode, "S": ret[:, 0], "E": ret[:, 1], "I": ret[:, 2], "R": ret[:, 3]}
    ode_data.set(ode_curr)

    # Gillespie
    # Passiamo gli stessi parametri usati per l'ODE e l'ABM
    # Nota: vax_pct per Gillespie segue la logica "lungo termine" (vaccinazione nuovi nati)
    vax_pct_gillespie = p["vax_pct"] if (is_long_term and p["vax_strat"] != "none") else 0.0

    loop = asyncio.get_running_loop()
    g_df = await loop.run_in_executor(
        None,
        run_gillespie_simulation,
        CORRECT_N,
        p["beta"],
        p["gamma"],
        sigma,
        steps,
        current_mu,  # <--- Nuovo parametro
        vax_pct_gillespie  # <--- Nuovo parametro
    )
    gillespie_data.set(g_df)

    # 3. ABM Loop
    speed_mode = p.get("speed_mode", "animata")

    if speed_mode == "turbo":
        status_msg.set("Esecuzione Turbo in corso...")

        # Eseguiamo il calcolo in un thread separato (ma bloccante per la logica sequenziale)
        # Per evitare di bloccare la UI totalmente su 5000 step, usiamo yield ogni tanto
        # Ma il modo più veloce in assoluto è fare il loop puro Python senza await.

        # Facciamo blocchi da 100 step per lasciare la UI reattiva al tasto "Stop" (se ci fosse)
        chunk_size = 100
        remaining_steps = steps

        while remaining_steps > 0:
            current_chunk = min(chunk_size, remaining_steps)
            for _ in range(current_chunk):
                model.step()

            remaining_steps -= current_chunk
            # Un micro await per non freezare il browser se fai 10.000 step
            await asyncio.sleep(0.001)

            # FINITO: Raccogli i dati UNA VOLTA SOLA alla fine
        model_instance.set(model)
        df = model.datacollector.get_model_vars_dataframe()
        run_data.set(df)

    else:
        # MODALITÀ CLASSICA (ANIMATA)
        # Aggiorna ogni singolo step (Lento ma vedi l'evoluzione)
        for i in range(steps):
            model.step()

            # Aggiornamento UI
            model_instance.set(model)
            df = model.datacollector.get_model_vars_dataframe()
            run_data.set(df)

            # Pausa estetica
            await asyncio.sleep(0.01)
    try:
        save_single_run_results(model, df, ode_curr, g_df)
        status_msg.set(f"Run salvata in {OUTPUT_DIR}")
    except Exception as e:
        status_msg.set(f" Errore salvataggio: {e}")

    is_running.set(False)


def run_batch_analysis():
    """Esegue batch analysis (Tab 2)."""
    if is_analyzing.value: return
    is_analyzing.set(True)
    stochastic_results.set(None)
    status_msg.set(f"Batch run ({stochastic_params.value['runs']} iterazioni)...")

    p = sim_params.value
    sp = stochastic_params.value
    peaks = []
    exceeded = 0

    for i in range(sp["runs"]):
        # Modello "leggero" per performance
        current_mu = p["mu"] if p["steps"] > LONG_TERM_THRESHOLD else 0.0

        m = VirusModel(
            N=p["N"], width=p["grid_size"], height=p["grid_size"],
            beta=p["beta"], gamma=p["gamma"], incubation_mean=p["incubation_mean"],
            topology=p["topology"], ws_k=p["ws_k"], ws_p=p["ws_p"], er_p=p["er_p"],
            comm_l=p["comm_l"], comm_k=p["comm_k"],
            vaccine_strategy=p["vax_strat"], vaccine_pct=p["vax_pct"],
            scheduler_type="random", prob_symptomatic=p["prob_symp"],
            mu=current_mu
        )

        local_peak = 0
        crossed = False
        for _ in range(p["steps"]):
            m.step()
            inf = sum(1 for a in m.agents if a.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC])
            if inf > local_peak: local_peak = inf
            if inf > sp["threshold"]: crossed = True

        peaks.append(local_peak)
        if crossed: exceeded += 1

    stochastic_results.set({"peaks": peaks, "probability": exceeded / sp["runs"]})
    is_analyzing.set(False)
    status_msg.set("Batch completato.")
    plot_path = save_batch_results_plot(peaks, threshold=sp["threshold"])

    status_msg.set(f"Batch salvato: {os.path.basename(plot_path)}")


def run_parameter_sweep():
    """Esegue parameter sweep (Tab 3)."""
    if is_sweeping.value: return
    is_sweeping.set(True)
    sweep_results.set(None)
    status_msg.set("Sweep in corso...")

    p = sim_params.value
    sw = sweep_params.value
    vals = np.linspace(sw["start"], sw["end"], sw["num_steps"])
    res = []

    for val in vals:
        curr_p = p.copy()
        curr_p[sw["parameter"]] = val
        is_long_term = curr_p["steps"] > LONG_TERM_THRESHOLD
        if is_long_term and curr_p["mu"] == 0.0:
            current_mu = 1.0 / (80 * 365.0)
        else:
            current_mu = curr_p["mu"] if is_long_term else 0.0

        m = VirusModel(
            N=curr_p["N"],
            width=curr_p["grid_size"],  # <--- Usa grid_size, non 20 fisso
            height=curr_p["grid_size"],  # <--- Usa grid_size
            beta=curr_p["beta"],
            gamma=curr_p["gamma"],
            incubation_mean=curr_p["incubation_mean"],
            topology=curr_p["topology"],
            ws_k=curr_p["ws_k"], ws_p=curr_p["ws_p"], er_p=curr_p["er_p"],
            ba_m=curr_p["ba_m"],  # <--- Aggiunto parametro mancante
            comm_l=curr_p["comm_l"], comm_k=curr_p["comm_k"],
            vaccine_strategy=curr_p["vax_strat"],
            vaccine_pct=curr_p["vax_pct"],
            scheduler_type="random",
            prob_symptomatic=curr_p["prob_symp"],
            # Parametri Lockdown (per coerenza)
            lockdown_threshold_pct=curr_p["lockdown_thresh"] if curr_p["lockdown_enabled"] else 1.0,
            lockdown_max_sd=curr_p["lockdown_max_sd"] if curr_p["lockdown_enabled"] else 0.0,
            mu=current_mu  # <--- FONDAMENTALE: passa il mu calcolato
        )
        peak = 0
        for _ in range(curr_p["steps"]):
            m.step()
            inf = sum(1 for a in m.agents if a.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC])
            if inf > peak: peak = inf
        res.append((val, peak))

    sweep_results.set(res)
    is_sweeping.set(False)
    status_msg.set("✅ Sweep completato.")
    plot_path = save_sweep_results_plot(res, sw["parameter"])

    status_msg.set(f"✅ Sweep salvato: {os.path.basename(plot_path)}")


# --- UI COMPONENTS ---

@solara.component
def SidebarParams():
    """Componente dedicato alla sidebar per pulizia codice."""
    busy = is_running.value or is_analyzing.value or is_sweeping.value
    p = sim_params.value

    def update(key, val):
        new_p = p.copy()
        new_p[key] = val
        sim_params.set(new_p)

    solara.Markdown("##  Configurazione")

    # 1. GENERALE (Sostituito ExpansionPanel con Details)
    with solara.Details("Generale", expand=True):
        solara.SliderInt("Popolazione", value=p["N"], min=50, max=10000, step=100, on_value=lambda v: update("N", v),
                         disabled=busy)
        solara.SliderInt("Step Simulazione", value=p["steps"], min=50, max=3650, step=50,
                     on_value=lambda v: update("steps", v), disabled=busy)

        # Mostra slider Mu solo se superiamo la soglia
        if p["steps"] > LONG_TERM_THRESHOLD:
            solara.Markdown("**Parametri Lungo Termine**")

            # Helper per convertire anni -> mu giornaliero
            def set_mu_from_years(years):
                val = 0.0 if years <= 0 else 1.0 / (years * 365.0)
                update("mu", val)

            # Calcola anni correnti da mu (inverso)
            current_years = int(1.0 / (p["mu"] * 365.0)) if p["mu"] > 0 else 80

            solara.SliderInt("Speranza di Vita (anni)", value=current_years, min=1, max=100,
                             on_value=set_mu_from_years, disabled=busy)
        solara.SliderInt("Griglia (LxL)", value=p["grid_size"], min=10, max=50, step=5,
                         on_value=lambda v: update("grid_size", v), disabled=busy)

    # 2. EPIDEMIOLOGIA
    with solara.Details("Epidemiologia", expand=False):
        solara.SliderFloat("Beta (Contagio)", value=p["beta"], min=0.1, max=1.0, step=0.05,
                           on_value=lambda v: update("beta", v), disabled=busy)
        solara.SliderFloat("Gamma (Guarigione)", value=p["gamma"], min=0.05, max=0.5, step=0.01,
                           on_value=lambda v: update("gamma", v), disabled=busy)
        solara.SliderFloat("Prob. Sintomi", value=p["prob_symp"], min=0.0, max=1.0, step=0.1,
                           on_value=lambda v: update("prob_symp", v), disabled=busy)
        solara.SliderInt("Incubazione Media (gg)", value=p["incubation_mean"], min=1, max=10,
                         on_value=lambda v: update("incubation_mean", v), disabled=busy)

    # 3. TOPOLOGIA
    with solara.Details("Rete e Topologia", expand=False):
        solara.Select("Tipo Topologia", value=p["topology"],
                      values=["grid", "network", "watts_strogatz", "erdos_renyi", "communities"],
                      on_value=lambda v: update("topology", v), disabled=busy)

        if p["topology"] == "watts_strogatz":
            solara.SliderInt("Vicini (k)", value=p["ws_k"], min=2, max=10, on_value=lambda v: update("ws_k", v),
                             disabled=busy)
            solara.SliderFloat("Prob. Rewire (p)", value=p["ws_p"], min=0.0, max=1.0, step=0.05,
                               on_value=lambda v: update("ws_p", v), disabled=busy)
        elif p["topology"] == "erdos_renyi":
            solara.SliderFloat("Prob. Link (p)", value=p["er_p"], min=0.0, max=0.2, step=0.01,
                               on_value=lambda v: update("er_p", v), disabled=busy)
        elif p["topology"] == "communities":
            solara.SliderInt("Num. Comunità", value=p["comm_l"], min=2, max=20, on_value=lambda v: update("comm_l", v),
                             disabled=busy)
            solara.SliderInt("Dim. Comunità", value=p["comm_k"], min=5, max=50, on_value=lambda v: update("comm_k", v),
                             disabled=busy)

    # 4. INTERVENTI
    with solara.Details("Interventi (Vaccini/Lockdown)", expand=False):
        solara.Select("Strategia Vaccini", value=p["vax_strat"], values=["none", "random", "targeted"],
                      on_value=lambda v: update("vax_strat", v), disabled=busy)
        if p["vax_strat"] != "none":
            solara.SliderFloat("% Vaccinati", value=p["vax_pct"], min=0.0, max=0.9, step=0.1,
                               on_value=lambda v: update("vax_pct", v), disabled=busy)

        solara.Markdown("---")
        solara.Checkbox(label="Attiva Lockdown Dinamico", value=p["lockdown_enabled"],
                        on_value=lambda v: update("lockdown_enabled", v), disabled=busy)
        if p["lockdown_enabled"]:
            solara.SliderFloat("Soglia Attivazione (% Infetti)", value=p["lockdown_thresh"], min=0.05, max=0.5,
                               step=0.05, on_value=lambda v: update("lockdown_thresh", v), disabled=busy)
            solara.SliderFloat("Max Distanziamento", value=p["lockdown_max_sd"], min=0.1, max=1.0, step=0.1,
                               on_value=lambda v: update("lockdown_max_sd", v), disabled=busy)
    solara.Select(
        label="Velocità Esecuzione",
        value=p.get("speed_mode", "animata"),
        values=["animata", "turbo"],
        on_value=lambda v: update("speed_mode", v),
        disabled=busy
    )
    if p.get("speed_mode") == "turbo":
        solara.Info("Turbo: Aggiorna il grafico solo alla fine. Molto veloce.", icon=False)


@solara.component
def Dashboard():
    busy = is_running.value or is_analyzing.value or is_sweeping.value

    with solara.Sidebar():
        SidebarParams()
        solara.Markdown("---")
        if status_msg.value:
            solara.Info(status_msg.value)

    with solara.Column(style={"padding": "20px", "max-width": "1200px", "margin": "0 auto"}):
        solara.Title("Simulatore CMCS")

        with solara.lab.Tabs():

            # --- TAB 1: LIVE ---
            with solara.lab.Tab("Simulazione & Curve"):
                with solara.Card():
                    solara.Button("Avvia Live Run", on_click=lambda: asyncio.create_task(run_live_simulation()),
                                  color="primary", disabled=busy, style={"width": "100%", "margin-bottom": "15px"})

                    if run_data.value is not None:
                        df = run_data.value
                        ode = ode_data.value
                        gsp = gillespie_data.value

                        # --- GRAFICO DELLE CURVE ---
                        fig, ax = plt.subplots(figsize=(10, 5))
                        N = model_instance.value.N

                        # 1. Modello ABM (Linee Solide)
                        ax.plot(df.index, df["S"] / N * 100, label="S (ABM)", color=AGENT_COLORS[0])
                        ax.plot(df.index, df["E"] / N * 100, label="E (ABM)", color=AGENT_COLORS[1])
                        ax.plot(df.index, (df["I_asymp"] + df["I_symp"]) / N * 100, label="I (ABM)", color=AGENT_COLORS[3])
                        ax.plot(df.index, df["R"] / N * 100, label="R (ABM)", color=AGENT_COLORS[4])

                        # 2. ODE (Tratteggiato)
                        if ode:
                            ax.plot(ode["t"], ode["S"] / N * 100, '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
                            ax.plot(ode["t"], ode["E"] / N * 100, '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
                            ax.plot(ode["t"], ode["I"] / N * 100, '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
                            ax.plot(ode["t"], ode["R"] / N * 100, '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")

                        # 3. Gillespie (Punteggiato)
                        if gsp is not None:
                            ax.plot(gsp["time"], gsp["S"] / N * 100, ':', color=AGENT_COLORS[0], alpha=0.9, label="S (Gillespie)")
                            ax.plot(gsp["time"], gsp["E"] / N * 100, ':', color=AGENT_COLORS[1], alpha=0.9, label="E (Gillespie)")
                            ax.plot(gsp["time"], gsp["I"] / N * 100, ':', color=AGENT_COLORS[3], alpha=0.9, label="I (Gillespie)")
                            ax.plot(gsp["time"], gsp["R"] / N * 100, ':', color=AGENT_COLORS[4], alpha=0.9, label="R (Gillespie)")


                        # 4. Lockdown Area
                        if "Lockdown" in df.columns:
                            active = df[df["Lockdown"] == 1]
                            if not active.empty:
                                ax.axvspan(active.index[0], active.index[-1], color='gray', alpha=0.2,
                                           label="Periodo Lockdown")

                        ax.set_title("Dinamica dell'Infezione (ABM vs ODE vs Gillespie)")
                        ax.set_xlabel("Giorni / Step")
                        ax.set_ylabel("Popolazione (%)")
                        ax.set_ylim(0, 100)
                        ax.legend(loc='upper right', fontsize='small', ncol=2)
                        ax.grid(True, alpha=0.3)
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)

                        # --- VISTE SPAZIALI ---
                        with solara.Row():
                            with solara.Column():
                                solara.Markdown("**Mappa Agenti**")
                                fig2, ax2 = plt.subplots(figsize=(5, 5))
                                model = model_instance.value
                                if model.G:
                                    pos = model.layout
                                    colors = [AGENT_COLORS[a.state] for a in model.agents]
                                    nx.draw_networkx_nodes(model.G, pos=pos, ax=ax2, node_size=30, node_color=colors)
                                    nx.draw_networkx_edges(model.G, pos=pos, ax=ax2, alpha=0.1)
                                else:
                                    grid = np.full((model.width, model.height), -1)
                                    for a in model.agents: grid[a.pos] = a.state
                                    ax2.imshow(grid, cmap=GRID_CMAP, vmin=-1, vmax=4)
                                ax2.axis('off')
                                solara.FigureMatplotlib(fig2)
                                plt.close(fig2)

                            with solara.Column():
                                solara.Markdown("**Rete di Petri (Stato Finale)**")
                                fig3, ax3 = plt.subplots(figsize=(5, 5))
                                last = df.iloc[-1]
                                draw_petri_net(ax3, last["S"], last["E"], last["I_asymp"] + last["I_symp"], last["R"])
                                solara.FigureMatplotlib(fig3)
                                plt.close(fig3)
                    else:
                        solara.Info("Configura i parametri a sinistra e premi 'Avvia Live Run'.")

            # --- TAB 2: BATCH ---
            with solara.lab.Tab("Analisi Batch"):
                with solara.Card():
                    with solara.Row():
                        solara.InputInt("Numero Run", value=stochastic_params.value["runs"],
                                        on_value=lambda v: stochastic_params.set(
                                            {**stochastic_params.value, "runs": v}))
                        solara.InputInt("Soglia Rischio", value=stochastic_params.value["threshold"],
                                        on_value=lambda v: stochastic_params.set(
                                            {**stochastic_params.value, "threshold": v}))
                    solara.Button("⚡ Avvia Batch", on_click=run_batch_analysis, color="warning", disabled=busy)

                    if stochastic_results.value:
                        res = stochastic_results.value
                        peaks = res["peaks"]
                        solara.Success(f"Probabilità superamento soglia: {res['probability'] * 100:.1f}%")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(peaks, bins=15, color="purple", alpha=0.7)
                        ax.axvline(stochastic_params.value['threshold'], color='r', linestyle='--')
                        ax.set_title("Distribuzione Picchi Epidemici")
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)

            # --- TAB 3: SWEEP ---
            with solara.lab.Tab("Sweep"):
                with solara.Card():
                    solara.Select("Parametro", value=sweep_params.value["parameter"],
                                  values=["beta", "gamma", "vax_pct", "prob_symp"],
                                  on_value=lambda v: sweep_params.set({**sweep_params.value, "parameter": v}))
                    with solara.Row():
                        solara.InputFloat("Min", value=sweep_params.value["start"],
                                          on_value=lambda v: sweep_params.set({**sweep_params.value, "start": v}))
                        solara.InputFloat("Max", value=sweep_params.value["end"],
                                          on_value=lambda v: sweep_params.set({**sweep_params.value, "end": v}))
                        solara.InputInt("Steps", value=sweep_params.value["num_steps"],
                                        on_value=lambda v: sweep_params.set({**sweep_params.value, "num_steps": v}))
                    solara.Button("🔍 Avvia Sweep", on_click=run_parameter_sweep, color="secondary", disabled=busy)

                    if sweep_results.value:
                        data = sweep_results.value
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot([x[0] for x in data], [x[1] for x in data], '-o', color='teal')
                        ax.set_xlabel(sweep_params.value["parameter"])
                        ax.set_ylabel("Picco Infetti")
                        ax.grid(True)
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)


@solara.component
def Page():
    solara.Title("Virus Sim")
    Dashboard()