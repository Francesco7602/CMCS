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
from .plotting import save_single_run_results, save_batch_results_plot

sim_params = solara.reactive({
    "N": 500,
    "steps": 100,
    "grid_size": 30,
    "beta": 0.6,
    "gamma": 0.1,
    "topology": "grid",
    "vax_strat": "none",
    "vax_pct": 0.0,
    "scheduler": "simultaneous",
    "prob_symp": 0.6,
})

model_instance = solara.reactive(None)
run_data = solara.reactive(None)
ode_data = solara.reactive(None)
batch_results = solara.reactive(None)
is_running = solara.reactive(False)
last_save_msg = solara.reactive("")


async def run_simulation():
    if is_running.value: return
    is_running.set(True)
    last_save_msg.set("")

    p = sim_params.value
    STEPS = p["steps"]
    INCUBATION_MEAN = 3

    # ODE
    sigma = 1.0 / INCUBATION_MEAN
    y0 = (p["N"] - 1, 1, 0, 0)
    t_ode = np.linspace(0, STEPS, STEPS)
    ret = odeint(seir_ode, y0, t_ode, args=(p["N"], p["beta"], sigma, p["gamma"]))
    ode_curr = {"t": t_ode, "S": ret[:, 0], "E": ret[:, 1], "I": ret[:, 2], "R": ret[:, 3]}
    ode_data.set(ode_curr)

    # Modello
    model = VirusModel(N=p["N"], width=p["grid_size"], height=p["grid_size"],
                       beta=p["beta"], gamma=p["gamma"], incubation_mean=INCUBATION_MEAN,
                       topology=p["topology"],
                       vaccine_strategy=p["vax_strat"],
                       vaccine_pct=p["vax_pct"],
                       scheduler_type=p["scheduler"],
                       prob_symptomatic=p["prob_symp"])

    model_instance.set(model)
    run_data.set(pd.DataFrame(columns=SHORT_LABELS + ["Lockdown"]))

    for i in range(STEPS):
        model.step()
        model_instance.set(model)
        run_data.set(model.datacollector.get_model_vars_dataframe())
        await asyncio.sleep(0.05)

    try:
        p1, p2 = save_single_run_results(model, run_data.value, ode_curr)
        last_save_msg.set(f"✅ Salvato con successo in {OUTPUT_DIR}")
    except Exception as e:
        last_save_msg.set(f"❌ Errore salvataggio: {e}")

    is_running.set(False)


def start_simulation_wrapper():
    asyncio.create_task(run_simulation())


def run_stochastic_batch():
    p = sim_params.value
    peak_infections = []
    last_save_msg.set("Esecuzione batch in corso...")

    for _ in range(30):
        m = VirusModel(N=p["N"], width=20, height=20,
                       beta=p["beta"], gamma=p["gamma"], incubation_mean=3,
                       topology="network",
                       vaccine_strategy="none", vaccine_pct=0.0,
                       scheduler_type="random", prob_symptomatic=0.6)
        history = []
        for _ in range(50):
            m.step()
            tot_inf = sum(1 for a in m.agents if a.state in [STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC])
            history.append(tot_inf)
        peak_infections.append(max(history))

    batch_results.set(peak_infections)

    try:
        pb = save_batch_results_plot(peak_infections)
        last_save_msg.set(f"✅ Batch salvato in {OUTPUT_DIR}")
    except Exception as e:
        last_save_msg.set(f"❌ Errore batch: {e}")


@solara.component
def Dashboard():
    with solara.Sidebar():
        solara.Markdown("## Parametri Simulazione")
        solara.SliderInt(label="Popolazione (N)", value=sim_params.value["N"], min=50, max=1000, step=50, on_value=lambda v: sim_params.set({**sim_params.value, "N": v}))
        solara.SliderInt(label="Durata (steps)", value=sim_params.value["steps"], min=20, max=500, step=10, on_value=lambda v: sim_params.set({**sim_params.value, "steps": v}))
        solara.SliderInt(label="Dimensione Griglia", value=sim_params.value["grid_size"], min=10, max=100, step=5, on_value=lambda v: sim_params.set({**sim_params.value, "grid_size": v}))
        solara.Select(label="Topologia", value=sim_params.value["topology"], values=["grid", "network"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "topology": v}))
        solara.Select(label="Scheduler", value=sim_params.value["scheduler"], values=["random", "simultaneous"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "scheduler": v}))
        solara.SliderFloat(label="Beta (Contagio)", value=sim_params.value["beta"], min=0.1, max=1.0, step=0.05,
                           on_value=lambda v: sim_params.set({**sim_params.value, "beta": v}))
        solara.SliderFloat(label="Gamma (Guarigione)", value=sim_params.value["gamma"], min=0.05, max=0.5, step=0.01,
                           on_value=lambda v: sim_params.set({**sim_params.value, "gamma": v}))

        solara.Markdown("---")
        solara.Markdown("### Vaccinazione")
        solara.Select(label="Metodo", value=sim_params.value["vax_strat"], values=["none", "random", "targeted"],
                      on_value=lambda v: sim_params.set({**sim_params.value, "vax_strat": v}))
        solara.SliderFloat(label="% Vaccinati", value=sim_params.value["vax_pct"], min=0.0, max=0.9, step=0.1,
                           on_value=lambda v: sim_params.set({**sim_params.value, "vax_pct": v}))

        solara.Markdown("---")
        solara.Button("Avvia Simulazione LIVE", on_click=start_simulation_wrapper, color="primary",
                      disabled=is_running.value, style={"width": "100%", "margin-bottom": "10px"})
        solara.Button("Analisi Batch (Estinzione)", on_click=run_stochastic_batch, color="warning",
                      style={"width": "100%"})

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

                    with solara.Row():
                        # --- GRAFICO 1 ---
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        if ode is not None:
                            ax1.plot(ode["t"], ode["S"], '--', color=AGENT_COLORS[0], alpha=0.4, label="S (ODE)")
                            ax1.plot(ode["t"], ode["E"], '--', color=AGENT_COLORS[1], alpha=0.4, label="E (ODE)")
                            ax1.plot(ode["t"], ode["I"], '--', color=AGENT_COLORS[3], alpha=0.4, label="I (ODE)")

                        if not df.empty:
                            ax1.plot(df["S"], label="S", color=AGENT_COLORS[0])
                            ax1.plot(df["E"], label="E", color=AGENT_COLORS[1])
                            ax1.plot(df["I_asymp"], label="I(hid)", color=AGENT_COLORS[2], linestyle="-.")
                            ax1.plot(df["I_symp"], label="I(det)", color=AGENT_COLORS[3])
                            ax1.plot(df["R"], label="R", color=AGENT_COLORS[4])

                            if "Lockdown" in df.columns:
                                lockdown_steps = df[df["Lockdown"] == 1].index
                                if len(lockdown_steps) > 0:
                                    ax1.axvspan(lockdown_steps[0], lockdown_steps[-1], color='red', alpha=0.1,
                                                label="Lockdown")

                        ax1.set_title(f"Step: {model.steps_count}")
                        ax1.set_xlim(0, sim_params.value["steps"])
                        ax1.set_ylim(0, model.N)
                        ax1.legend(loc="upper right", fontsize='x-small', ncol=2)
                        ax1.grid(True, alpha=0.3)
                        solara.FigureMatplotlib(fig1)

                        # --- GRAFICO 2 ---
                        fig2, ax2 = plt.subplots(figsize=(6, 4))

                        if model.topology == "network":
                            colors = [AGENT_COLORS[a.state] for a in model.agents]
                            pos = nx.spring_layout(model.G, seed=42)
                            nx.draw(model.G, pos=pos, ax=ax2, node_size=30, node_color=colors, width=0.2)
                            ax2.set_title("Diffusione Rete")
                        else:
                            # Inizializziamo con STATE_EMPTY (-1)
                            grid_arr = np.full((model.grid.width, model.grid.height), STATE_EMPTY)
                            for a in model.agents:
                                grid_arr[a.pos] = a.state

                            # vmin=-1 forza il bianco per lo sfondo
                            ax2.imshow(grid_arr, cmap=GRID_CMAP, vmin=-1, vmax=4, interpolation="nearest")
                            ax2.set_title("Diffusione Griglia")

                            legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in
                                               zip(AGENT_COLORS, SHORT_LABELS)]
                            ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1),
                                       fontsize='x-small')

                        ax2.axis('off')
                        solara.FigureMatplotlib(fig2)
                else:
                    solara.Info("Premi START per vedere l'animazione.")

            # --- TAB 2: BATCH ---
            with solara.lab.Tab("Analisi Stocastica"):
                if batch_results.value is not None:
                    peaks = batch_results.value
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.hist(peaks, bins=15, color="purple", alpha=0.7, edgecolor='black')
                    ax3.set_title(f"Distribuzione Picchi Epidemici (su {len(peaks)} runs)")
                    ax3.set_xlabel("Picco Massimo Infetti")
                    ax3.set_ylabel("Frequenza")
                    ax3.grid(axis='y', alpha=0.5, linestyle='--')
                    solara.FigureMatplotlib(fig3)
                else:
                    solara.Info("Premi 'Analisi Batch' per generare l'istogramma.")


@solara.component
def Page():
    Dashboard()
