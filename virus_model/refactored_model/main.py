# virus_model/refactored_model/main.py
#python -m solara run virus_model.refactored_model.main
import solara
import solara.lab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import asyncio
from scipy.integrate import odeint
import networkx as nx

from .constants import (
    STATE_EMPTY,
    SHORT_LABELS,
    AGENT_COLORS,
    GRID_CMAP,
    OUTPUT_DIR,
    STATE_INFECTED_ASYMPTOMATIC,
    STATE_INFECTED_SYMPTOMATIC,
)
from .ode import seir_ode
from .model import VirusModel
from .plotting import save_single_run_results, save_batch_results_plot, draw_petri_net
from .gillespie import run_gillespie_simulation

sim_params = solara.reactive({
    "N": 500,
    "steps": 100,
    "grid_size": 30,
    "beta": 0.6,
    "gamma": 0.1,
    "topology": "grid",
    "ws_k": 4,
    "ws_p": 0.1,
    "er_p": 0.1,
    "comm_l": 5,
    "comm_k": 20,
    "vax_strat": "none",
    "vax_pct": 0.0,
    "scheduler": "simultaneous",
    "prob_symp": 0.6,
})

model_instance = solara.reactive(None)
run_data = solara.reactive(None)
ode_data = solara.reactive(None)
gillespie_data = solara.reactive(None)
stochastic_results = solara.reactive(None)

is_running = solara.reactive(False)
is_analyzing_reachability = solara.reactive(False)
is_sweeping = solara.reactive(False)
last_save_msg = solara.reactive("")

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
sweep_results = solara.reactive(None)


async def run_simulation():
    if is_running.value: return
    is_running.set(True)
    last_save_msg.set("")

    p = sim_params.value
    STEPS = p["steps"]
    INCUBATION_MEAN = 3
    sigma = 1.0 / INCUBATION_MEAN


    # Modello ABM
    model = VirusModel(N=p["N"], width=p["grid_size"], height=p["grid_size"],
                       beta=p["beta"], gamma=p["gamma"], incubation_mean=INCUBATION_MEAN,
                       topology=p["topology"],
                       ws_k=p["ws_k"], ws_p=p["ws_p"], er_p=p["er_p"],
                       comm_l=p["comm_l"], comm_k=p["comm_k"],
                       vaccine_strategy=p["vax_strat"],
                       vaccine_pct=p["vax_pct"],
                       scheduler_type=p["scheduler"],
                       prob_symptomatic=p["prob_symp"])
    
    model_instance.set(model)
    
    # ODE & Gillespie con N corretto dal modello
    CORRECT_N = model.N
    y0 = (CORRECT_N - 1, 1, 0, 0)
    t_ode = np.linspace(0, STEPS, STEPS)
    ret = odeint(seir_ode, y0, t_ode, args=(CORRECT_N, p["beta"], sigma, p["gamma"]))
    ode_curr = {"t": t_ode, "S": ret[:, 0], "E": ret[:, 1], "I": ret[:, 2], "R": ret[:, 3]}
    ode_data.set(ode_curr)
    
    loop = asyncio.get_running_loop()
    gillespie_df = await loop.run_in_executor(None, run_gillespie_simulation, CORRECT_N, p["beta"], p["gamma"], sigma, STEPS)
    gillespie_data.set(gillespie_df)

    run_data.set(pd.DataFrame(columns=SHORT_LABELS + ["Lockdown"]))

    for i in range(STEPS):
        model.step()
        model_instance.set(model)
        run_data.set(model.datacollector.get_model_vars_dataframe())
        await asyncio.sleep(0.05)

    try:
        save_single_run_results(model, run_data.value, ode_curr, gillespie_data.value)
        last_save_msg.set(f"✅ Report salvato con successo in {OUTPUT_DIR}")
    except Exception as e:
        last_save_msg.set(f"❌ Errore salvataggio: {e}")

    is_running.set(False)


def start_simulation_wrapper():
    asyncio.create_task(run_simulation())


def run_reachability_analysis():
    if is_analyzing_reachability.value: return
    is_analyzing_reachability.set(True)
    stochastic_results.set(None)
    
    sim_p = sim_params.value
    stoch_p = stochastic_params.value
    
    peak_infections = []
    runs_exceeding_threshold = 0
    
    last_save_msg.set(f"Esecuzione di {stoch_p['runs']} run in corso...")

    for i in range(stoch_p["runs"]):
        m = VirusModel(N=sim_p["N"], width=20, height=20,
                       beta=sim_p["beta"], gamma=sim_p["gamma"], incubation_mean=3,
                       topology=sim_p["topology"],
                       ws_k=sim_p["ws_k"], ws_p=sim_p["ws_p"], er_p=sim_p["er_p"],
                       comm_l=sim_p["comm_l"], comm_k=sim_p["comm_k"],
                       vaccine_strategy=sim_p["vax_strat"], vaccine_pct=sim_p["vax_pct"],
                       scheduler_type="random", prob_symptomatic=sim_p["prob_symp"])
        
        history = []
        exceeded = False
        for _ in range(sim_p["steps"]):
            m.step()
            tot_inf = sum(1 for a in m.agents if a.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC])
            history.append(tot_inf)
            if not exceeded and tot_inf > stoch_p["threshold"]:
                exceeded = True
        
        if exceeded:
            runs_exceeding_threshold += 1
        
        peak_infections.append(max(history) if history else 0)

    probability = runs_exceeding_threshold / stoch_p["runs"]
    stochastic_results.set({"probability": probability, "peaks": peak_infections})

    try:
        save_batch_results_plot(peak_infections)
        last_save_msg.set(f"✅ Batch salvato in {OUTPUT_DIR}")
    except Exception as e:
        last_save_msg.set(f"❌ Errore batch: {e}")
    
    is_analyzing_reachability.set(False)


def run_parameter_sweep():
    if is_sweeping.value: return
    is_sweeping.set(True)
    sweep_results.set(None)
    last_save_msg.set("")

    sim_p = sim_params.value
    sw_p = sweep_params.value
    
    param_to_sweep = sw_p["parameter"]
    param_values = np.linspace(sw_p["start"], sw_p["end"], sw_p["num_steps"])
    
    results = []
    
    last_save_msg.set(f"Esecuzione sweep per '{param_to_sweep}'...")

    for i, val in enumerate(param_values):
        last_save_msg.set(f"Sweep per '{param_to_sweep}': run {i + 1}/{sw_p['num_steps']}")
        
        current_sim_params = sim_p.copy()
        current_sim_params[param_to_sweep] = val
        
        m = VirusModel(N=current_sim_params["N"], width=20, height=20,
                       beta=current_sim_params["beta"], gamma=current_sim_params["gamma"], incubation_mean=3,
                       topology=current_sim_params["topology"],
                       ws_k=current_sim_params["ws_k"], ws_p=current_sim_params["ws_p"], er_p=current_sim_params["er_p"],
                       comm_l=current_sim_params["comm_l"], comm_k=current_sim_params["comm_k"],
                       vaccine_strategy=current_sim_params["vax_strat"], vaccine_pct=current_sim_params["vax_pct"],
                       scheduler_type="random", prob_symptomatic=current_sim_params["prob_symp"])
        
        peak_infection = 0
        for _ in range(current_sim_params["steps"]):
            m.step()
            total_infected = sum(1 for a in m.agents if a.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC])
            if total_infected > peak_infection:
                peak_infection = total_infected
        
        results.append((val, peak_infection))

    sweep_results.set(results)
    last_save_msg.set("✅ Sweep completato.")
    is_sweeping.set(False)


@solara.component
def Dashboard():
    # Workaround to ensure slider updates when N changes
    stochastic_params.use()
    sim_params.use()
    
    is_busy = is_running.value or is_analyzing_reachability.value or is_sweeping.value

    with solara.Sidebar():
        solara.Markdown("## Parametri Simulazione")

        is_communities = sim_params.value["topology"] == "communities"
        
        solara.SliderInt(label="Popolazione (N)", value=sim_params.value["N"], min=50, max=1000, step=50, 
                        on_value=lambda v: sim_params.set({**sim_params.value, "N": v}),
                        disabled=is_communities or is_busy)
        
        solara.SliderInt(label="Durata (steps)", value=sim_params.value["steps"], min=20, max=500, step=10, on_value=lambda v: sim_params.set({**sim_params.value, "steps": v}), disabled=is_busy)
        solara.SliderInt(label="Dimensione Griglia", value=sim_params.value["grid_size"], min=10, max=100, step=5, on_value=lambda v: sim_params.set({**sim_params.value, "grid_size": v}), disabled=is_busy)
        
        solara.Select(label="Topologia", value=sim_params.value["topology"], values=["grid", "network", "watts_strogatz", "erdos_renyi", "communities"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "topology": v}), disabled=is_busy)
        
        if sim_params.value["topology"] == "watts_strogatz":
            solara.SliderInt(label="Watts-Strogatz K", value=sim_params.value["ws_k"], min=2, max=20, on_value=lambda v: sim_params.set({**sim_params.value, "ws_k": v}), disabled=is_busy)
            solara.SliderFloat(label="Watts-Strogatz P", value=sim_params.value["ws_p"], min=0.0, max=1.0, step=0.05, on_value=lambda v: sim_params.set({**sim_params.value, "ws_p": v}), disabled=is_busy)

        if sim_params.value["topology"] == "erdos_renyi":
            solara.SliderFloat(label="Erdos-Renyi P", value=sim_params.value["er_p"], min=0.0, max=0.2, step=0.01, on_value=lambda v: sim_params.set({**sim_params.value, "er_p": v}), disabled=is_busy)

        if is_communities:
            solara.SliderInt(label="Numero Comunità (L)", value=sim_params.value["comm_l"], min=2, max=50, on_value=lambda v: sim_params.set({**sim_params.value, "comm_l": v}), disabled=is_busy)
            solara.SliderInt(label="Dimensione Comunità (K)", value=sim_params.value["comm_k"], min=5, max=50, on_value=lambda v: sim_params.set({**sim_params.value, "comm_k": v}), disabled=is_busy)
            total_n = sim_params.value["comm_l"] * sim_params.value["comm_k"]
            solara.Info(f"Popolazione totale (N = L*K): {total_n}")

        solara.Select(label="Scheduler", value=sim_params.value["scheduler"], values=["random", "simultaneous"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "scheduler": v}), disabled=is_busy)
        solara.SliderFloat(label="Beta (Contagio)", value=sim_params.value["beta"], min=0.1, max=1.0, step=0.05,
                           on_value=lambda v: sim_params.set({**sim_params.value, "beta": v}), disabled=is_busy)
        solara.SliderFloat(label="Gamma (Guarigione)", value=sim_params.value["gamma"], min=0.05, max=0.5, step=0.01,
                           on_value=lambda v: sim_params.set({**sim_params.value, "gamma": v}), disabled=is_busy)

        solara.Markdown("---")
        solara.Markdown("### Vaccinazione")
        solara.Select(label="Metodo", value=sim_params.value["vax_strat"], values=["none", "random", "targeted"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "vax_strat": v}), disabled=is_busy)
        solara.SliderFloat(label="% Vaccinati", value=sim_params.value["vax_pct"], min=0.0, max=0.9, step=0.1,
                           on_value=lambda v: sim_params.set({**sim_params.value, "vax_pct": v}), disabled=is_busy)

        solara.Markdown("---")
        solara.Button("Avvia Simulazione LIVE", on_click=start_simulation_wrapper, color="primary",
                      disabled=is_busy, style={"width": "100%", "margin-bottom": "10px"})
        solara.Button("Avvia Analisi di Raggiungibilità", on_click=run_reachability_analysis, color="warning",
                      disabled=is_busy, style={"width": "100%"})

        if last_save_msg.value:
            solara.Success(last_save_msg.value)

    with solara.Column(style={"padding": "20px"}):
        solara.Markdown("# Simulatore Epidemiologico")

        with solara.lab.Tabs():
            # --- TAB 1: LIVE ---
            with solara.lab.Tab("Simulazione Live"):
                if model_instance.value is not None and run_data.value is not None:
                    df = run_data.value
                    model = model_instance.value
                    ode = ode_data.value
                    gsp = gillespie_data.value

                    # --- GRAFICO 1 (CURVE) ---
                    fig1, ax1 = plt.subplots(figsize=(12, 4))
                    if ode is not None:
                        ax1.plot(ode["t"], ode["S"], '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
                        ax1.plot(ode["t"], ode["E"], '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
                        ax1.plot(ode["t"], ode["I"], '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
                        ax1.plot(ode["t"], ode["R"], '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")
                    
                    if gsp is not None:
                        ax1.plot(gsp["time"], gsp["S"], ':', color=AGENT_COLORS[0], alpha=0.9, label="S (Gillespie)")
                        ax1.plot(gsp["time"], gsp["E"], ':', color=AGENT_COLORS[1], alpha=0.9, label="E (Gillespie)")
                        ax1.plot(gsp["time"], gsp["I"], ':', color=AGENT_COLORS[3], alpha=0.9, label="I (Gillespie)")
                        ax1.plot(gsp["time"], gsp["R"], ':', color=AGENT_COLORS[4], alpha=0.9, label="R (Gillespie)")

                    if not df.empty:
                        ax1.plot(df["S"], label="S (ABM)", color=AGENT_COLORS[0])
                        ax1.plot(df["E"], label="E (ABM)", color=AGENT_COLORS[1])
                        ax1.plot(df["I_asymp"] + df["I_symp"], label="I (ABM)", color=AGENT_COLORS[3])
                        ax1.plot(df["R"], label="R (ABM)", color=AGENT_COLORS[4])

                        if "Lockdown" in df.columns:
                            lockdown_steps = df[df["Lockdown"] == 1].index
                            if len(lockdown_steps) > 0:
                                ax1.axvspan(lockdown_steps[0], lockdown_steps[-1], color='red', alpha=0.1,
                                            label="Lockdown")

                    ax1.set_title(f"Dinamica SEIR - Step: {model.steps_count}")
                    ax1.set_xlim(0, sim_params.value["steps"])
                    ax1.set_ylim(0, model.N)
                    ax1.legend(loc="upper right", fontsize='x-small', ncol=3)
                    ax1.grid(True, alpha=0.3)
                    solara.FigureMatplotlib(fig1, dependencies=[df])

                    with solara.Row():
                        # --- GRAFICO 2 (GRIGLIA/RETE) ---
                        with solara.Column():
                            fig2, ax2 = plt.subplots(figsize=(6, 5))
                            if model.G is not None:
                                colors = [AGENT_COLORS[a.state] for a in model.agents]
                                pos = nx.spring_layout(model.G, seed=42)
                                nx.draw(model.G, pos=pos, ax=ax2, node_size=30, node_color=colors, width=0.2)
                                ax2.set_title("Diffusione Rete")
                            else:
                                grid_arr = np.full((model.grid.width, model.grid.height), STATE_EMPTY)
                                for a in model.agents:
                                    grid_arr[a.pos] = a.state
                                ax2.imshow(grid_arr, cmap=GRID_CMAP, vmin=-1, vmax=4, interpolation="nearest")
                                ax2.set_title("Diffusione Griglia")
                                legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in
                                                   zip(AGENT_COLORS, SHORT_LABELS)]
                                ax2.legend(handles=legend_elements, loc='upper left',
                                           fontsize='x-small')
                            ax2.axis('off')
                            solara.FigureMatplotlib(fig2, dependencies=[df])

                        # --- GRAFICO 3 (RETE DI PETRI) ---
                        with solara.Column():
                            fig3, ax3 = plt.subplots(figsize=(6, 5))
                            if not df.empty:
                                latest = df.iloc[-1]
                                S, E, I, R = latest["S"], latest["E"], latest["I_asymp"] + latest["I_symp"], latest["R"]
                                draw_petri_net(ax3, S, E, I, R)
                            else:
                                # Draw empty petri net before simulation starts
                                draw_petri_net(ax3, sim_params.value['N']-1, 1, 0, 0)
                            solara.FigureMatplotlib(fig3, dependencies=[df])
                else:
                    solara.Info("Premi START per vedere l'animazione.")

            # --- TAB 2: BATCH ---
            with solara.lab.Tab("Analisi Stocastica"):
                solara.Markdown("### Impostazioni Analisi di Raggiungibilità")
                solara.SliderInt(
                    label="Numero di Run", 
                    value=stochastic_params.value["runs"], 
                    min=10, max=200, step=10, 
                    on_value=lambda v: stochastic_params.set({**stochastic_params.value, "runs": v}),
                    disabled=is_busy
                )
                solara.SliderInt(
                    label="Soglia Infetti (Threshold)", 
                    value=stochastic_params.value["threshold"], 
                    min=1, max=sim_params.value["N"], 
                    on_value=lambda v: stochastic_params.set({**stochastic_params.value, "threshold": v}),
                    disabled=is_busy
                )
                
                if stochastic_results.value is not None:
                    res = stochastic_results.value
                    prob_pct = res["probability"] * 100
                    
                    solara.Success(
                        f"Probabilità che gli infetti superino la soglia ({stochastic_params.value['threshold']}): "
                        f"**{prob_pct:.1f}%** (calcolata su {stochastic_params.value['runs']} run)"
                    )

                    peaks = res["peaks"]
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.hist(peaks, bins=15, color="purple", alpha=0.7, edgecolor='black')
                    ax3.axvline(x=stochastic_params.value['threshold'], color='r', linestyle='--', linewidth=2, label=f"Soglia ({stochastic_params.value['threshold']})")
                    ax3.set_title(f"Distribuzione Picchi Epidemici (su {len(peaks)} runs)")
                    ax3.set_xlabel("Picco Massimo Infetti")
                    ax3.set_ylabel("Frequenza")
                    ax3.legend()
                    ax3.grid(axis='y', alpha=0.5, linestyle='--')
                    solara.FigureMatplotlib(fig3)
                else:
                    solara.Info("Premi 'Avvia Analisi di Raggiungibilità' per generare il calcolo di probabilità e l'istogramma.")

            # --- TAB 3: SWEEP ---
            with solara.lab.Tab("Parameter Sweep"):
                solara.Markdown("### Impostazioni Parameter Sweep")
                
                solara.Select(
                    label="Parametro da Variare",
                    value=sweep_params.value["parameter"],
                    values=["beta", "gamma", "prob_symp", "vax_pct"],
                    on_value=lambda v: sweep_params.set({**sweep_params.value, "parameter": v}),
                    disabled=is_busy
                )

                solara.SliderFloat(
                    label="Valore Iniziale",
                    value=sweep_params.value["start"],
                    min=0.0, max=1.0, step=0.05,
                    on_value=lambda v: sweep_params.set({**sweep_params.value, "start": v}),
                    disabled=is_busy
                )
                solara.SliderFloat(
                    label="Valore Finale",
                    value=sweep_params.value["end"],
                    min=0.0, max=1.0, step=0.05,
                    on_value=lambda v: sweep_params.set({**sweep_params.value, "end": v}),
                    disabled=is_busy
                )
                solara.SliderInt(
                    label="Numero di Step",
                    value=sweep_params.value["num_steps"],
                    min=5, max=50,
                    on_value=lambda v: sweep_params.set({**sweep_params.value, "num_steps": v}),
                    disabled=is_busy
                )
                
                solara.Button("Avvia Parameter Sweep", on_click=run_parameter_sweep, color="secondary",
                              disabled=is_busy, style={"width": "100%", "margin-bottom": "10px"})

                if sweep_results.value:
                    res = sweep_results.value
                    param_name = sweep_params.value["parameter"]
                    
                    x_values = [r[0] for r in res]
                    y_values = [r[1] for r in res]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_values, y_values, marker='o', linestyle='-', color='teal')
                    ax.set_title(f"Impatto di '{param_name.capitalize()}' sul Picco di Infezioni", pad=20)
                    ax.set_xlabel(f"Valore di {param_name.capitalize()}")
                    ax.set_ylabel("Picco Massimo di Infetti")
                    ax.grid(True, alpha=0.5, linestyle='--')
                    ax.set_ylim(bottom=0)
                    solara.FigureMatplotlib(fig)
                else:
                    solara.Info("Premi 'Avvia Parameter Sweep' per generare il grafico di analisi.", margin=2)


@solara.component
def Page():
    Dashboard()
