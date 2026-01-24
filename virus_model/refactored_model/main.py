# virus_model/refactored_model/main.py
# python -m solara run virus_model.refactored_model.main
import solara
import solara.lab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import threading

from mesa.visualization import SolaraViz
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.model import Model
from mesa.visualization.utils import update_counter

from .constants import (
    AGENT_COLORS,
    STATE_INFECTED_ASYMPTOMATIC,
    STATE_INFECTED_SYMPTOMATIC,
)
from .ode import seir_ode
from .model import VirusModel
from .plotting import draw_petri_net, save_batch_results_plot
from .gillespie import run_gillespie_simulation

# --------------------------------------------------------------------------------
# --- Modello Esteso per Integrazione con Mesa/Solara ---
# --------------------------------------------------------------------------------

class SolaraVirusModel(VirusModel):
    """
    Un'estensione del modello VirusModel per integrarsi meglio con SolaraViz.
    Aggiunge il calcolo dei dati di confronto (ODE/Gillespie) e gestisce
    il ciclo di vita della simulazione.
    """
    def __init__(self, max_steps=100, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.ode_data = None
        self.gillespie_data = None
        self.gillespie_thread = None

    @property
    def steps(self):
        """
        Returns the current step count from the scheduler.
        Returns 0 if the scheduler has not been initialized yet.
        """
        if hasattr(self, "schedule"):
            return self.schedule.steps
        return 0

    @steps.setter
    def steps(self, value):
        """
        Setter for the 'steps' property.
        This is required to avoid an AttributeError during model initialization,
        as the parent mesa.Model class attempts to set self.steps = 0.
        We can safely ignore the value being set.
        """
        pass

    def step(self):
        """Avanza il modello di uno step e gestisce la fine della simulazione."""
        if self.steps_count >= self.max_steps:
            self.running = False
            return
        
        # Al primo step, avvia i calcoli di confronto
        if self.steps_count == 1:
            self._calculate_comparison_data()
            
        super().step()

    def _calculate_comparison_data(self):
        """Calcola i dati per ODE e (in un thread separato) per Gillespie."""
        CORRECT_N = self.N
        y0 = (CORRECT_N - 1, 1, 0, 0)
        t_ode = np.linspace(0, self.max_steps, self.max_steps)
        sigma = 1.0 / self.incubation_mean
        
        # ODE (veloce, eseguito direttamente)
        ret = odeint(seir_ode, y0, t_ode, args=(CORRECT_N, self.beta, sigma, self.gamma))
        self.ode_data = {"t": t_ode, "S": ret[:, 0], "E": ret[:, 1], "I": ret[:, 2], "R": ret[:, 3]}
        
        # Gillespie (lento, eseguito in un thread per non bloccare l'UI)
        self.gillespie_thread = threading.Thread(
            target=self._run_gillespie_in_background,
            args=(CORRECT_N, self.beta, self.gamma, sigma, self.max_steps)
        )
        self.gillespie_thread.start()

    def _run_gillespie_in_background(self, *args):
        """Wrapper per eseguire Gillespie e salvare il risultato."""
        self.gillespie_data = run_gillespie_simulation(*args)

# --------------------------------------------------------------------------------
# --- Configurazione Visualizzazione Mesa ---
# --------------------------------------------------------------------------------

def agent_portrayal(agent):
    """Specifica come disegnare ogni agente sulla griglia."""
    if agent is None:
        return
    return {
        "color": AGENT_COLORS[agent.state],
        "size": 50,
        "marker": "o",
    }

model_params = {
    "N": {"type": "SliderInt", "value": 500, "min": 50, "max": 1000, "step": 50, "label": "Popolazione (N)"},
    "max_steps": {"type": "SliderInt", "value": 100, "min": 20, "max": 500, "step": 10, "label": "Durata (steps)"},
    "width": {"type": "SliderInt", "value": 30, "min": 10, "max": 100, "step": 5, "label": "Grid Width"},
    "height": {"type": "SliderInt", "value": 30, "min": 10, "max": 100, "step": 5, "label": "Grid Height"},
    "beta": {"type": "SliderFloat", "value": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "label": "Beta (Contagio)"},
    "gamma": {"type": "SliderFloat", "value": 0.1, "min": 0.05, "max": 0.5, "step": 0.01, "label": "Gamma (Guarigione)"},
    "incubation_mean": {"type": "SliderInt", "value": 3, "min": 1, "max": 10, "label": "Periodo Incubazione"},
    "prob_symptomatic": {"type": "SliderFloat", "value": 0.6, "min": 0.0, "max": 1.0, "step": 0.1, "label": "Prob. Sintomi"},
    "topology": {"type": "Select", "value": "grid", "values": ["grid", "network", "watts_strogatz", "erdos_renyi", "communities"], "label": "Topologia"},
    "ws_k": {"type": "SliderInt", "value": 4, "min": 2, "max": 20, "label": "Watts-Strogatz K"},
    "ws_p": {"type": "SliderFloat", "value": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Watts-Strogatz P"},
    "er_p": {"type": "SliderFloat", "value": 0.1, "min": 0.0, "max": 0.2, "step": 0.01, "label": "Erdos-Renyi P"},
    "ba_m": {"type": "SliderInt", "value": 2, "min": 1, "max": 10, "label": "Barabasi-Albert M"},
    "comm_l": {"type": "SliderInt", "value": 5, "min": 2, "max": 50, "label": "Num Comunità (L)"},
    "comm_k": {"type": "SliderInt", "value": 20, "min": 5, "max": 50, "label": "Dim Comunità (K)"},
    "vaccine_strategy": {"type": "Select", "value": "none", "values": ["none", "random", "targeted"], "label": "Strategia Vaccino"},
    "vaccine_pct": {"type": "SliderFloat", "value": 0.0, "min": 0.0, "max": 0.9, "step": 0.1, "label": "% Vaccinati"},
    "scheduler_type": {"type": "Select", "value": "simultaneous", "values": ["random", "simultaneous"], "label": "Scheduler"},
}

# --------------------------------------------------------------------------------
# --- Componenti di Visualizzazione Custom ---
# --------------------------------------------------------------------------------

@solara.component
def seir_curves_plot(model: Model, steps: int):
    """Componente per visualizzare le curve SEIR (ABM, ODE, Gillespie)."""
    update_counter.get() # Ensure reactivity
    
    fig, ax = plt.subplots(figsize=(12, 4))
    df = model.datacollector.get_model_vars_dataframe()
    
    if model.ode_data:
        ode = model.ode_data
        ax.plot(ode["t"], ode["S"], '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
        ax.plot(ode["t"], ode["E"], '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
        ax.plot(ode["t"], ode["I"], '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
        ax.plot(ode["t"], ode["R"], '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")

    if model.gillespie_data is not None:
        gsp = model.gillespie_data
        ax.plot(gsp["time"], gsp["S"], ':', color=AGENT_COLORS[0], alpha=0.9, label="S (Gillespie)")
        ax.plot(gsp["time"], gsp["E"], ':', color=AGENT_COLORS[1], alpha=0.9, label="E (Gillespie)")
        ax.plot(gsp["time"], gsp["I"], ':', color=AGENT_COLORS[3], alpha=0.9, label="I (Gillespie)")
        ax.plot(gsp["time"], gsp["R"], ':', color=AGENT_COLORS[4], alpha=0.9, label="R (Gillespie)")
    elif model.steps_count > 1:
        ax.text(0.5, 0.5, 'Gillespie data is loading...', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if not df.empty:
        ax.plot(df["S"], label="S (ABM)", color=AGENT_COLORS[0])
        ax.plot(df["E"], label="E (ABM)", color=AGENT_COLORS[1])
        ax.plot(df["I_asymp"] + df["I_symp"], label="I (ABM)", color=AGENT_COLORS[3])
        ax.plot(df["R"], label="R (ABM)", color=AGENT_COLORS[4])

    ax.set_title(f"Dinamica SEIR - Step: {model.steps_count}")
    ax.set_xlim(0, model.max_steps)
    ax.set_ylim(0, model.N)
    
    # Solo se ci sono etichette, crea la legenda
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize='x-small', ncol=3)
        
    ax.grid(True, alpha=0.3)

    solara.FigureMatplotlib(fig)
    plt.close(fig)

@solara.component
def petri_net_plot(model: Model, steps: int):
    """Componente per visualizzare la Rete di Petri dello stato attuale."""
    update_counter.get() # Ensure reactivity
    
    fig, ax = plt.subplots(figsize=(6, 5))
    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        latest = df.iloc[-1]
        S, E, I, R = latest["S"], latest["E"], latest["I_asymp"] + latest["I_symp"], latest["R"]
        draw_petri_net(ax, S, E, I, R)
    else:
        if "N" in model_params:
            draw_petri_net(ax, model_params["N"]["value"] - 1, 1, 0, 0)
    
    solara.FigureMatplotlib(fig)
    plt.close(fig)

space_component = make_mpl_space_component(agent_portrayal=agent_portrayal)


@solara.component
def BottomPlots(model: Model, steps: int):
    with solara.Columns([1, 1]):
        space_component(model=model)
        petri_net_plot(model=model, steps=steps)

@solara.component
def AllPlots(model: Model):
    with solara.Column(style={"width": "100%"}):
        seir_curves_plot(model, model.steps)
        BottomPlots(model, model.steps)

# --------------------------------------------------------------------------------
# --- Pagina Principale dell'Applicazione ---
# --------------------------------------------------------------------------------

# @solara.component
# def StochasticAnalysisTab(): ... # Lasciato per riferimento futuro

# @solara.component
# def ParameterSweepTab(): ... # Lasciato per riferimento futuro

@solara.component
def Page():
    """Componente principale che assembla l'interfaccia utente."""
    solara.Markdown("# Simulatore Epidemiologico con Mesa e Solara")

    # Crea un'istanza iniziale del modello da passare a SolaraViz.
    # Questo assicura che i componenti ricevano un'istanza valida fin da subito.
    default_params = {name: options["value"] for name, options in model_params.items()}
    initial_model = SolaraVirusModel(**default_params)

    # Creiamo i componenti di visualizzazione usando gli helper di Mesa
    
    SolaraViz(
        initial_model,
        model_params=model_params,
        components=[
            (AllPlots, 0),
        ],
        name="Simulatore Epidemiologico",
    )
