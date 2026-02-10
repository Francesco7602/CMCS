# virus_model/refactored_model/main.py
"""
This is the main application file for the CMCS-Sim (Complex Model for Contagion Simulation).

It uses the Solara framework to create a web-based dashboard for running and visualizing
virus spread simulations. The dashboard supports three main modes of operation:
1.  **Live Simulation**: A single, animated run of the Agent-Based Model (ABM) compared
    against deterministic Ordinary Differential Equations (ODE) and a stochastic
    Gillespie simulation.
2.  **Batch Analysis**: Runs the ABM multiple times to analyze the distribution of
    outcomes (e.g., epidemic peaks) and calculate probabilities.
3.  **Parameter Sweep**: Systematically varies a chosen parameter to observe its
    impact on the epidemic peak, averaging results over several runs per data point.

The application is structured with a sidebar for parameter configuration and a tabbed
main area for displaying results.
"""
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
from mesa.batchrunner import batch_run

from .constants import (
    STATE_EMPTY, SHORT_LABELS, AGENT_COLORS, GRID_CMAP, OUTPUT_DIR,
    STATE_INFECTED_ASYMPTOMATIC, STATE_INFECTED_SYMPTOMATIC, MU_GROUP,
    CHILD, TEEN, ADULT, SENIOR,
)

# Calculate average disease mortality rate based on the model's age distribution.
# This is used for the ODE and Gillespie models to align with the ABM's demographics.
AGE_DISTRIBUTION_WEIGHTS = [0.1, 0.1, 0.6, 0.2]
AVG_MU_DISEASE = sum(
    MU_GROUP[i] * AGE_DISTRIBUTION_WEIGHTS[i]
    for i in [CHILD, TEEN, ADULT, SENIOR]
)
from .ode import seir_ode
from .model import VirusModel
from .plotting import save_single_run_results, save_batch_results_plot, draw_petri_net, save_sweep_results_plot
from .gillespie import run_gillespie_simulation


class _VirusModelWrapper(VirusModel):
    """
    A wrapper for the VirusModel to make it compatible with Mesa's `batch_run`.

    Mesa's `batch_run` can pass an extra 'rng' argument which is not expected
    by the base VirusModel constructor. This wrapper intercepts and removes it.
    """
    def __init__(self, **kwargs):
        kwargs.pop('rng', None)
        super().__init__(**kwargs)


# Threshold in days to switch between short-term and long-term simulation dynamics (e.g., enabling vital dynamics).
LONG_TERM_THRESHOLD = 365

# --- Reactive State Management using Solara ---

# All simulation parameters, configurable from the UI.
sim_params = solara.reactive({
    "N": 500,
    "steps": 100,
    "grid_size": 30,
    # Epidemiology
    "beta": 0.6,
    "gamma": 0.1,
    "prob_symp": 0.6,
    "incubation_mean": 3,
    "mu": 0.0,
    # Topology
    "topology": "grid",
    "ws_k": 4, "ws_p": 0.1,
    "er_p": 0.1,
    "ba_m": 2,
    "comm_l": 5, "comm_k": 20,
    # Interventions
    "vax_strat": "none",
    "vax_pct": 0.0,
    "lockdown_enabled": False,
    "lockdown_thresh": 0.2,
    "lockdown_max_sd": 0.8,
    # System
    "scheduler": "simultaneous",
    "speed_mode": "animata",  # Options: "animata", "turbo"
})

# Parameters for the stochastic batch analysis.
stochastic_params = solara.reactive({
    "runs": 50,
    "threshold": 150,
})

# Parameters for the parameter sweep analysis.
sweep_params = solara.reactive({
    "parameter": "beta",
    "start": 0.1,
    "end": 1.0,
    "num_steps": 10,
})

# --- Result and State Variables ---

# Holds the model instance during a live run.
model_instance = solara.reactive(None)
# Data from the latest ABM run.
run_data = solara.reactive(None)
# Data from the corresponding ODE model.
ode_data = solara.reactive(None)
# Data from the corresponding Gillespie simulation.
gillespie_data = solara.reactive(None)
# Results from the batch analysis.
stochastic_results = solara.reactive(None)
# Results from the parameter sweep.
sweep_results = solara.reactive(None)

# UI state flags to manage button disabling and status messages.
is_running = solara.reactive(False)
is_analyzing = solara.reactive(False)
is_sweeping = solara.reactive(False)
status_msg = solara.reactive("")


# --- SIMULATION LOGIC ---

async def run_live_simulation():
    """
    Executes a single, live simulation run (Tab 1).

    This function orchestrates the setup and execution of three different models
    for comparison: the main Agent-Based Model (ABM), a deterministic SEIR ODE model,
    and a stochastic Gillespie simulation. It updates the UI progressively for
    animated runs or at the end for "turbo" runs.
    """
    if is_running.value: return
    is_running.set(True)
    status_msg.set("")

    p = sim_params.value
    steps = p["steps"]
    sigma = 1.0 / p["incubation_mean"]

    # --- Long-term simulation logic ---
    is_long_term = steps > LONG_TERM_THRESHOLD

    # If it's a long-term run but natural death rate (mu) is 0, set a default.
    if is_long_term and p["mu"] == 0.0:
        current_mu = 1.0 / (80 * 365.0)  # Default lifespan of 80 years
    else:
        current_mu = p["mu"] if is_long_term else 0.0

    # 1. Initialize ABM (passing the dynamic mu)
    model = VirusModel(
        N=p["N"], width=p["grid_size"], height=p["grid_size"],
        beta=p["beta"], gamma=p["gamma"], incubation_mean=p["incubation_mean"],
        topology=p["topology"],
        ws_k=p["ws_k"], ws_p=p["ws_p"], er_p=p["er_p"], ba_m=p["ba_m"],
        comm_l=p["comm_l"], comm_k=p["comm_k"],
        vaccine_strategy=p["vax_strat"], vaccine_pct=p["vax_pct"],
        scheduler_type=p["scheduler"], prob_symptomatic=p["prob_symp"],
        lockdown_threshold_pct=p["lockdown_thresh"] if p["lockdown_enabled"] else 1.0,
        lockdown_max_sd=p["lockdown_max_sd"] if p["lockdown_enabled"] else 0.0,
        lockdown_active_threshold=0.05,
        mu=current_mu
    )
    model_instance.set(model)

    # 2. Run ODE & Gillespie models in the background
    CORRECT_N = model.N
    t_ode = np.linspace(0, steps, steps)
    p_lock_value = 1.0 - p["lockdown_max_sd"]

    # Calculate initial conditions (static vaccination at time 0)
    initial_vaccinated = int(CORRECT_N * p["vax_pct"]) if p["vax_strat"] != "none" else 0
    y0 = (CORRECT_N - 1 - initial_vaccinated, 1, 0, initial_vaccinated, 0)  # Aggiunto lo 0 per D
    #y0 = (CORRECT_N - 1 - initial_vaccinated, 1, 0, initial_vaccinated) # S, E, I, R

    # Vaccination parameter for ODE (for newborns)
    # If short-term (mu=0), vax_pct_ode must be 0 (no births).
    # If long-term (mu>0), vax_pct_ode is the vaccination percentage.
    vax_pct_ode = p["vax_pct"] if (is_long_term and p["vax_strat"] != "none") else 0.0

    # Run ODE solver
    ret = odeint(seir_ode, y0, t_ode, args=(
        CORRECT_N, p["beta"], sigma, p["gamma"], current_mu, AVG_MU_DISEASE, vax_pct_ode,
        p["lockdown_enabled"], p["lockdown_thresh"], p_lock_value
    ))
    ode_curr = {"t": t_ode, "S": ret[:, 0], "E": ret[:, 1], "I": ret[:, 2], "R": ret[:, 3], "Deaths": ret[:, 4]}
    ode_data.set(ode_curr)

    # Run Gillespie simulation
    vax_pct_gillespie = p["vax_pct"] if (is_long_term and p["vax_strat"] != "none") else 0.0
    loop = asyncio.get_running_loop()
    g_df, g_births = await loop.run_in_executor(
        None, run_gillespie_simulation, CORRECT_N, p["beta"], p["gamma"], sigma, steps,
        current_mu, AVG_MU_DISEASE, vax_pct_gillespie,
        p["lockdown_enabled"], p["lockdown_thresh"], p_lock_value
    )
    gillespie_data.set(g_df)

    # 3. ABM Execution Loop
    speed_mode = p.get("speed_mode", "animata")

    if speed_mode == "turbo":
        status_msg.set("Running in Turbo Mode...")
        # Run the simulation in chunks to keep the UI responsive.
        chunk_size = 100
        remaining_steps = steps
        while remaining_steps > 0:
            current_chunk = min(chunk_size, remaining_steps)
            for _ in range(current_chunk):
                model.step()
            remaining_steps -= current_chunk
            await asyncio.sleep(0.001) # Yield to event loop
        
        # Collect data once at the end
        model_instance.set(model)
        df = model.datacollector.get_model_vars_dataframe()
        run_data.set(df)
    else: # Animated mode
        for i in range(steps):
            model.step()
            # Update UI at each step
            model_instance.set(model)
            df = model.datacollector.get_model_vars_dataframe()
            run_data.set(df)
            await asyncio.sleep(0.01) # Aesthetic pause

    # Save results to files
    try:
        save_single_run_results(model, run_data.value, ode_curr, g_df)
        
        # Calculate and display mortality percentages for all models
        total_population = model.total_agents_created
        #total_population = model.N # N from ABM (used as reference population)
        
        abm_mortality_msg = ""
        if not df.empty and "Deaths" in df.columns and total_population > 0:
            total_abm_deaths = df["Deaths"].iloc[-1]
            abm_mortality_percentage = (total_abm_deaths / total_population) * 100
            abm_mortality_msg = f"ABM: {abm_mortality_percentage:.2f}%"

        ode_mortality_msg = ""
        if ode_curr and "Deaths" in ode_curr and len(ode_curr["Deaths"]) > 0 and total_population > 0:
            ode_total_participants = model.N + (current_mu * model.N * steps)
            total_ode_deaths = ode_curr["Deaths"][-1]
            ode_mortality_percentage = (total_ode_deaths / ode_total_participants) * 100
            ode_mortality_msg = f"ODE: {ode_mortality_percentage:.2f}%"
        
        gillespie_mortality_msg = ""
        if g_df is not None and not g_df.empty and "Deaths" in g_df.columns and total_population > 0:
            g_total_participants = CORRECT_N + g_births
            total_gillespie_deaths = g_df["Deaths"].iloc[-1]
            gillespie_mortality_percentage = (total_gillespie_deaths / g_total_participants) * 100
            gillespie_mortality_msg = f"Gillespie: {gillespie_mortality_percentage:.2f}%"

        mortality_messages = [msg for msg in [abm_mortality_msg, ode_mortality_msg, gillespie_mortality_msg] if msg]
        
        if mortality_messages:
            combined_mortality_msg = "Mortality: " + ", ".join(mortality_messages)
            status_msg.set(f"{combined_mortality_msg} | Run results saved in {OUTPUT_DIR}")
        else:
            status_msg.set(f"Run results saved in {OUTPUT_DIR}")
    except Exception as e:
        status_msg.set(f"Error saving results: {e}")

    is_running.set(False)


async def run_batch_analysis():
    """
    Executes a batch analysis (Tab 2) using `mesa.batch_run`.

    This function runs the ABM multiple times with the same set of parameters to
    analyze the stochastic nature of the model. It calculates the distribution
    of epidemic peaks and the probability of exceeding a user-defined risk threshold.
    """
    if is_analyzing.value: return
    is_analyzing.set(True)
    stochastic_results.set(None)
    status_msg.set(f"Starting batch run ({stochastic_params.value['runs']} iterations)...")

    p = sim_params.value
    sp = stochastic_params.value

    current_mu = p["mu"] if p["steps"] > LONG_TERM_THRESHOLD else 0.0

    model_params = {
        "N": p["N"], "width": p["grid_size"], "height": p["grid_size"],
        "beta": p["beta"], "gamma": p["gamma"], "incubation_mean": p["incubation_mean"],
        "topology": p["topology"], "ws_k": p["ws_k"], "ws_p": p["ws_p"], "er_p": p["er_p"], "ba_m": p["ba_m"],
        "comm_l": p["comm_l"], "comm_k": p["comm_k"],
        "vaccine_strategy": p["vax_strat"], "vaccine_pct": p["vax_pct"],
        "scheduler_type": p["scheduler"], "prob_symptomatic": p["prob_symp"],
        "mu": current_mu,
        "lockdown_threshold_pct": p["lockdown_thresh"] if p["lockdown_enabled"] else 1.0,
        "lockdown_max_sd": p["lockdown_max_sd"] if p["lockdown_enabled"] else 0.0,
        "lockdown_active_threshold": 0.05,
    }

    # Run mesa's batch_run in a separate thread to avoid blocking the UI
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, batch_run, _VirusModelWrapper, model_params,
        None, sp["runs"], 1, p["steps"], True
    )

    results_df = pd.DataFrame(results)
    peaks = []
    exceeded = 0

    # Process results to find the peak infected count for each run
    if "RunId" in results_df.columns:
        for _, run_df in results_df.groupby("RunId"):
            infected = run_df["I_asymp"] + run_df["I_symp"]
            local_peak = infected.max()
            peaks.append(local_peak)
            if local_peak > sp["threshold"]:
                exceeded += 1
    elif not results_df.empty:
        infected = results_df["I_asymp"] + results_df["I_symp"]
        local_peak = infected.max()
        peaks.append(local_peak)
        if local_peak > sp["threshold"]:
            exceeded += 1

    stochastic_results.set({"peaks": peaks, "probability": exceeded / sp["runs"] if sp["runs"] > 0 else 0})
    is_analyzing.set(False)
    status_msg.set("Batch analysis complete.")

    # Calculate average mortality
    avg_mortality_msg = ""
    if "RunId" in results_df.columns and not results_df.empty and "TotalAgents" in results_df.columns:
        final_deaths_per_run = results_df.groupby("RunId")["Deaths"].last()
        final_agents_per_run = results_df.groupby("RunId")["TotalAgents"].last()
        avg_deaths = final_deaths_per_run.mean()
        avg_total_agents = final_agents_per_run.mean()
        total_population = p["N"]
        if total_population > 0:
            avg_mortality_percentage = (avg_deaths / avg_total_agents) * 100
            avg_mortality_msg = f" | Avg. Mortality: {avg_mortality_percentage:.2f}%"

    plot_path = save_batch_results_plot(peaks, threshold=sp["threshold"])
    status_msg.set(f"Batch plot saved: {os.path.basename(plot_path)}{avg_mortality_msg}")


async def run_parameter_sweep():
    """
    Executes a parameter sweep (Tab 3) using `mesa.batch_run`.

    This function systematically varies one parameter across a defined range. For each
    parameter value, it runs the simulation multiple times (e.g., 10) and calculates
    the average epidemic peak. This helps in understanding the sensitivity of the
    model to different parameters.
    """
    if is_sweeping.value: return
    is_sweeping.set(True)
    sweep_results.set(None)
    status_msg.set("Sweep in progress (10 runs per data point)...")

    p = sim_params.value
    sw = sweep_params.value
    vals = np.linspace(sw["start"], sw["end"], sw["num_steps"])

    is_long_term = p["steps"] > LONG_TERM_THRESHOLD
    current_mu = (1.0 / (80 * 365.0)) if (is_long_term and p["mu"] == 0.0) else (p["mu"] if is_long_term else 0.0)

    model_params = {
        "N": p["N"], "width": p["grid_size"], "height": p["grid_size"],
        "beta": p["beta"], "gamma": p["gamma"], "incubation_mean": p["incubation_mean"],
        "topology": p["topology"], "ws_k": p["ws_k"], "ws_p": p["ws_p"], "er_p": p["er_p"], "ba_m": p["ba_m"],
        "comm_l": p["comm_l"], "comm_k": p["comm_k"],
        "vaccine_strategy": p["vax_strat"],
        "vaccine_pct": p["vax_pct"],
        "scheduler_type": p["scheduler"],
        "prob_symptomatic": p["prob_symp"],
        "mu": current_mu,
        "lockdown_threshold_pct": p["lockdown_thresh"] if p["lockdown_enabled"] else 1.0,
        "lockdown_max_sd": p["lockdown_max_sd"] if p["lockdown_enabled"] else 0.0,
        "lockdown_active_threshold": 0.05,
    }

    # Map from short UI parameter names to longer model parameter names
    param_map = {
        "vax_pct": "vaccine_pct",
        "prob_symp": "prob_symptomatic",
        "lockdown_thresh": "lockdown_threshold_pct",
    }
    sweep_model_param = param_map.get(sw["parameter"], sw["parameter"])
    model_params[sweep_model_param] = vals

    # Run the sweep in a separate thread
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, batch_run, _VirusModelWrapper, model_params,
        None, 10, 1, p["steps"], True
    )

    results_df = pd.DataFrame(results)

    # Process sweep results
    res_df = pd.DataFrame()
    if sweep_model_param in results_df.columns and "TotalAgents" in results_df.columns:
        # Group by the swept parameter and iteration to process each run individually
        grouped = results_df.groupby([sweep_model_param, "iteration"])
        
        # Calculate the peak infected count for each individual run
        peaks_per_run = grouped.apply(lambda run_df: (run_df["I_asymp"] + run_df["I_symp"]).max())
        
        # Calculate the final death count for each individual run
        deaths_per_run = grouped["Deaths"].last()
        total_agents_per_run = grouped["TotalAgents"].last()

        # For each parameter value, calculate the mean of the peaks and deaths
        avg_peaks = peaks_per_run.groupby(sweep_model_param).mean()
        avg_deaths = deaths_per_run.groupby(sweep_model_param).mean()
        avg_total_agents = total_agents_per_run.groupby(sweep_model_param).mean()  # Media del denominatore

        # Create a combined DataFrame with the results, ensuring the correct column name
        res_df = pd.DataFrame({
            sw["parameter"]: avg_peaks.index,
            'avg_peak': avg_peaks.values,
            'avg_deaths': avg_deaths.values,
            'avg_total_agents': avg_total_agents.values
        })

        res_df['mortality_pct'] = (res_df['avg_deaths'] / res_df['avg_total_agents']) * 100

    sweep_results.set(res_df)
    is_sweeping.set(False)
    status_msg.set("Parameter sweep complete.")

    if not res_df.empty:
        plot_path = save_sweep_results_plot(res_df, sw["parameter"])
        status_msg.set(f"Sweep plot saved: {os.path.basename(plot_path)}")


# --- UI COMPONENTS ---

@solara.component
def SidebarParams():
    """
    A Solara component for the sidebar, containing all simulation parameters.
    """
    busy = is_running.value or is_analyzing.value or is_sweeping.value
    p = sim_params.value

    def update(key, val):
        new_p = p.copy()
        new_p[key] = val
        sim_params.set(new_p)

    solara.Markdown("##  Configuration")

    # 1. General Parameters
    with solara.Details("General", expand=True):
        solara.SliderInt("Population", value=p["N"], min=50, max=10000, step=100, on_value=lambda v: update("N", v), disabled=busy)
        solara.SliderInt("Simulation Steps", value=p["steps"], min=50, max=3650, step=50, on_value=lambda v: update("steps", v), disabled=busy)

        if p["steps"] > LONG_TERM_THRESHOLD:
            solara.Markdown("**Long-Term Parameters**")
            def set_mu_from_years(years):
                val = 0.0 if years <= 0 else 1.0 / (years * 365.0)
                update("mu", val)
            current_years = int(1.0 / (p["mu"] * 365.0)) if p["mu"] > 0 else 80
            solara.SliderInt("Life Expectancy (years)", value=current_years, min=1, max=100, on_value=set_mu_from_years, disabled=busy)
        
        solara.SliderInt("Grid Size (LxL)", value=p["grid_size"], min=10, max=50, step=5, on_value=lambda v: update("grid_size", v), disabled=busy)

    # 2. Epidemiology Parameters
    with solara.Details("Epidemiology", expand=False):
        solara.SliderFloat("Beta (Transmission)", value=p["beta"], min=0.1, max=1.0, step=0.05, on_value=lambda v: update("beta", v), disabled=busy)
        solara.SliderFloat("Gamma (Recovery)", value=p["gamma"], min=0.05, max=0.5, step=0.01, on_value=lambda v: update("gamma", v), disabled=busy)
        solara.SliderFloat("Symptom Probability", value=p["prob_symp"], min=0.0, max=1.0, step=0.1, on_value=lambda v: update("prob_symp", v), disabled=busy)
        solara.SliderInt("Avg. Incubation (days)", value=p["incubation_mean"], min=1, max=10, on_value=lambda v: update("incubation_mean", v), disabled=busy)

    # 3. Topology Parameters
    with solara.Details("Network & Topology", expand=False):
        solara.Select("Topology Type", value=p["topology"], values=["grid", "network", "watts_strogatz", "erdos_renyi", "communities"], on_value=lambda v: update("topology", v), disabled=busy)

        if p["topology"] == "watts_strogatz":
            solara.SliderInt("Neighbors (k)", value=p["ws_k"], min=2, max=10, on_value=lambda v: update("ws_k", v), disabled=busy)
            solara.SliderFloat("Rewire Prob (p)", value=p["ws_p"], min=0.0, max=1.0, step=0.05, on_value=lambda v: update("ws_p", v), disabled=busy)
        elif p["topology"] == "erdos_renyi":
            solara.SliderFloat("Link Prob (p)", value=p["er_p"], min=0.0, max=0.2, step=0.01, on_value=lambda v: update("er_p", v), disabled=busy)
        elif p["topology"] == "communities":
            solara.SliderInt("Num. Communities", value=p["comm_l"], min=2, max=20, on_value=lambda v: update("comm_l", v), disabled=busy)
            solara.SliderInt("Community Size", value=p["comm_k"], min=5, max=50, on_value=lambda v: update("comm_k", v), disabled=busy)

    # 4. Intervention Parameters
    with solara.Details("Interventions (Vaccines/Lockdown)", expand=False):
        solara.Select("Vaccine Strategy", value=p["vax_strat"], values=["none", "random", "targeted"], on_value=lambda v: update("vax_strat", v), disabled=busy)
        if p["vax_strat"] != "none":
            solara.SliderFloat("% Vaccinated", value=p["vax_pct"], min=0.0, max=0.9, step=0.1, on_value=lambda v: update("vax_pct", v), disabled=busy)

        solara.Markdown("---")
        solara.Checkbox(label="Enable Dynamic Lockdown", value=p["lockdown_enabled"], on_value=lambda v: update("lockdown_enabled", v), disabled=busy)
        if p["lockdown_enabled"]:
            solara.SliderFloat("Activation Threshold (% Infected)", value=p["lockdown_thresh"], min=0.05, max=0.5, step=0.05, on_value=lambda v: update("lockdown_thresh", v), disabled=busy)
            solara.SliderFloat("Max Social Distancing", value=p["lockdown_max_sd"], min=0.1, max=1.0, step=0.1, on_value=lambda v: update("lockdown_max_sd", v), disabled=busy)
    
    solara.Select(label="Execution Speed", value=p.get("speed_mode", "animata"), values=["animata", "turbo"], on_value=lambda v: update("speed_mode", v), disabled=busy)
    if p.get("speed_mode") == "turbo":
        solara.Info("Turbo mode only updates the plot at the end.", icon=False)

    solara.Select(label="Activation Scheduler", value=p.get("scheduler", "simultaneous"),
                  values=[("Synchronous (Simultaneous)", "simultaneous"), ("Uniform (Ordered)", "uniform"), ("Random (Shuffled)", "random"), ("Poisson (Random Interval)", "poisson")],
                  on_value=lambda v: update("scheduler", v), disabled=busy)


@solara.component
def Dashboard():
    """
    The main dashboard component, which lays out the sidebar and the tabbed interface.
    """
    busy = is_running.value or is_analyzing.value or is_sweeping.value

    with solara.Sidebar():
        SidebarParams()
        solara.Markdown("---")
        if status_msg.value:
            solara.Info(status_msg.value)

    with solara.Column(style={"padding": "20px", "max-width": "1200px", "margin": "0 auto"}):
        solara.Title("CMCS Simulator")

        with solara.lab.Tabs():

            # --- TAB 1: LIVE SIMULATION ---
            with solara.lab.Tab("Simulation & Curves"):
                with solara.Card():
                    solara.Button("Start Live Run", on_click=lambda: asyncio.create_task(run_live_simulation()), color="primary", disabled=busy, style={"width": "100%", "margin-bottom": "15px"})

                    if run_data.value is not None:

                        df = run_data.value
                        ode = ode_data.value
                        gsp = gillespie_data.value

                        # --- Epidemic Curves Plot ---
                        fig, ax = plt.subplots(figsize=(10, 5))
                        N = model_instance.value.N

                        # 1. ABM Model (Solid lines)
                        ax.plot(df.index, df["S"] / N * 100, label="S (ABM)", color=AGENT_COLORS[0])
                        ax.plot(df.index, df["E"] / N * 100, label="E (ABM)", color=AGENT_COLORS[1])
                        ax.plot(df.index, (df["I_asymp"] + df["I_symp"]) / N * 100, label="I (ABM)", color=AGENT_COLORS[3])
                        ax.plot(df.index, df["R"] / N * 100, label="R (ABM)", color=AGENT_COLORS[4])

                        # 2. ODE Model (Dashed lines)
                        if ode:
                            ax.plot(ode["t"], ode["S"] / N * 100, '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
                            ax.plot(ode["t"], ode["E"] / N * 100, '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
                            ax.plot(ode["t"], ode["I"] / N * 100, '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
                            ax.plot(ode["t"], ode["R"] / N * 100, '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")

                        # 3. Gillespie Model (Dotted lines)
                        if gsp is not None:
                            ax.plot(gsp["time"], gsp["S"] / N * 100, ':', color=AGENT_COLORS[0], alpha=0.9, label="S (Gillespie)")
                            ax.plot(gsp["time"], gsp["E"] / N * 100, ':', color=AGENT_COLORS[1], alpha=0.9, label="E (Gillespie)")
                            ax.plot(gsp["time"], gsp["I"] / N * 100, ':', color=AGENT_COLORS[3], alpha=0.9, label="I (Gillespie)")
                            ax.plot(gsp["time"], gsp["R"] / N * 100, ':', color=AGENT_COLORS[4], alpha=0.9, label="R (Gillespie)")

                        # 4. Lockdown Period Visualization
                        if "Lockdown" in df.columns:
                            active = df[df["Lockdown"] == 1]
                            if not active.empty:
                                ax.axvspan(active.index[0], active.index[-1], color='gray', alpha=0.2, label="Lockdown Period")

                        ax.set_title("Infection Dynamics (ABM vs ODE vs Gillespie)")
                        ax.set_xlabel("Days / Steps")
                        ax.set_ylabel("Population (%)")
                        ax.set_ylim(0, 100)
                        ax.legend(loc='upper right', fontsize='small', ncol=2)
                        ax.grid(True, alpha=0.3)
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)
                        # --- Spatial Views (Grid/Network and Transition Schema) ---
                        with solara.Row():
                            with solara.Column():
                                solara.Markdown("**Agent Map**")
                                fig2, ax2 = plt.subplots(figsize=(5, 5))
                                model = model_instance.value
                                if model.G: # Network topology
                                    pos = model.layout
                                    colors = [AGENT_COLORS[a.state] for a in model.agents]
                                    nx.draw_networkx_nodes(model.G, pos=pos, ax=ax2, node_size=30, node_color=colors)
                                    nx.draw_networkx_edges(model.G, pos=pos, ax=ax2, alpha=0.1)
                                else: # Grid topology
                                    grid = np.full((model.width, model.height), -1)
                                    for a in model.agents: grid[a.pos] = a.state
                                    ax2.imshow(grid, cmap=GRID_CMAP, vmin=-1, vmax=4)
                                ax2.axis('off')
                                solara.FigureMatplotlib(fig2)
                                plt.close(fig2)

                            with solara.Column():
                                solara.Markdown("**Transition Schema**")
                                fig3, ax3 = plt.subplots(figsize=(5, 5))
                                last = df.iloc[-1]
                                draw_petri_net(ax3, last["S"], last["E"], last["I_asymp"], last["I_symp"], last["R"])
                                solara.FigureMatplotlib(fig3)
                                plt.close(fig3)
                    else:
                        solara.Info("Configure parameters on the left and click 'Start Live Run'.")

            # --- TAB 2: BATCH ANALYSIS ---
            with solara.lab.Tab("Batch Analysis"):
                with solara.Card():
                    with solara.Row():
                        solara.InputInt("Number of Runs", value=stochastic_params.value["runs"], on_value=lambda v: stochastic_params.set({**stochastic_params.value, "runs": v}))
                        solara.InputInt("Risk Threshold", value=stochastic_params.value["threshold"], on_value=lambda v: stochastic_params.set({**stochastic_params.value, "threshold": v}))
                    solara.Button("Start Batch Analysis", on_click=lambda: asyncio.create_task(run_batch_analysis()), color="warning", disabled=busy)

                    if stochastic_results.value:
                        res = stochastic_results.value
                        peaks = res["peaks"]
                        solara.Success(f"Probability of exceeding threshold: {res['probability'] * 100:.1f}%")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(peaks, bins=15, color="purple", alpha=0.7)
                        ax.axvline(stochastic_params.value['threshold'], color='r', linestyle='--')
                        ax.set_title("Distribution of Epidemic Peaks")
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)

            # --- TAB 3: PARAMETER SWEEP ---
            with solara.lab.Tab("Parameter Sweep"):
                with solara.Card():
                    def update_sweep_params(key, value):
                        sweep_params.set({**sweep_params.value, key: value})
                        sweep_results.set(None)

                    solara.Select("Parameter", value=sweep_params.value["parameter"],
                                  values=["beta", "gamma", "vax_pct", "prob_symp", "lockdown_thresh"],
                                  on_value=lambda v: update_sweep_params("parameter", v))
                    with solara.Row():
                        solara.InputFloat("Min", value=sweep_params.value["start"], on_value=lambda v: update_sweep_params("start", v))
                        solara.InputFloat("Max", value=sweep_params.value["end"], on_value=lambda v: update_sweep_params("end", v))
                        solara.InputInt("Steps", value=sweep_params.value["num_steps"], on_value=lambda v: update_sweep_params("num_steps", v))
                    solara.Button("Start Parameter Sweep", on_click=lambda: asyncio.create_task(run_parameter_sweep()), color="secondary", disabled=busy)

                    if sweep_results.value is not None and not sweep_results.value.empty:
                        data_df = sweep_results.value
                        param_name = sweep_params.value["parameter"]
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Primary axis for infection peak
                        color1 = 'teal'
                        ax.set_xlabel(param_name, fontsize=12)
                        ax.set_ylabel("Average Infection Peak (Max)", color=color1, fontsize=12)
                        ax.plot(data_df[param_name], data_df['avg_peak'], '-o', color=color1, label='Avg. Infection Peak')
                        ax.tick_params(axis='y', labelcolor=color1)
                        ax.grid(True, linestyle='--', alpha=0.6)

                        # Secondary axis for mortality
                        ax2 = ax.twinx()
                        color2 = 'crimson'
                        ax2.set_ylabel("Average Mortality (%)", color=color2, fontsize=12)
                        ax2.plot(data_df[param_name], data_df['mortality_pct'], '-o', color=color2, label='Avg. Mortality (%)')
                        ax2.tick_params(axis='y', labelcolor=color2)
                        
                        ax.set_title(f"Parameter Sweep for '{param_name}'", fontsize=14)
                        fig.tight_layout()
                        
                        solara.FigureMatplotlib(fig)
                        plt.close(fig)


@solara.component
def Page():
    """The main page component that sets the title and renders the Dashboard."""
    solara.Title("Virus Sim")
    Dashboard()