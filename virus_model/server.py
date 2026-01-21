import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import odeint  # Per la matematica (ODE)
from matplotlib.colors import ListedColormap
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# =========================
# CONFIGURAZIONE COLORI E PARAMETRI
# =========================

# Colori per i grafici (S, E, I, R)
PLOT_COLORS = ["tab:blue", "gold", "tab:red", "tab:gray"]
STATE_LABELS = ["Susceptible", "Exposed", "Infected", "Recovered"]

# Colori per la Griglia (0=Bianco/Vuoto, 1..4=Stati)
GRID_COLORS = ["white"] + PLOT_COLORS
GRID_CMAP = ListedColormap(GRID_COLORS)
STATE_TO_INT = {"Susceptible": 1, "Exposed": 2, "Infected": 3, "Recovered": 4}


# =========================
# 1. MODELLO MATEMATICO (MACRO - ODE)
# =========================
def seir_ode(y, t, N, beta, sigma, gamma):
    """
    Equazioni differenziali per il confronto teorico.
    S' = -beta * S * I / N
    E' = beta * S * I / N - sigma * E
    I' = sigma * E - gamma * I
    R' = gamma * I
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# =========================
# 2. AGENTE (MICRO)
# =========================
class VirusAgent(Agent):
    def __init__(self, model, beta, gamma, incubation_mean):
        super().__init__(model)
        self.state = "Susceptible"
        self.beta = beta
        self.gamma = gamma
        self.infection_time = None
        # Variazione casuale del tempo di incubazione per realismo
        self.incubation_period = max(1, int(self.random.gauss(incubation_mean, 2)))

    def step(self):
        # Se il lockdown è attivo, la probabilità di muoversi crolla
        if self.random.random() > self.model.social_distancing:
            self.move()

        self.progress_state()

        if self.state == "Infected":
            self.infect_neighbors()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if possible_steps:
            self.model.grid.move_agent(self, self.random.choice(possible_steps))

    def infect_neighbors(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        for n in neighbors:
            if n.state == "Susceptible" and self.random.random() < self.beta:
                n.state = "Exposed"
                n.infection_time = self.model.steps

    def progress_state(self):
        # Transizione Exposed -> Infected
        if self.state == "Exposed" and self.infection_time is not None:
            if self.model.steps - self.infection_time >= self.incubation_period:
                self.state = "Infected"
        # Transizione Infected -> Recovered
        elif self.state == "Infected" and self.random.random() < self.gamma:
            self.state = "Recovered"


# =========================
# 3. MODELLO (AMBIENTE)
# =========================
class VirusModel(Model):
    def __init__(self, N=800, width=50, height=50,
                 beta=0.5, gamma=0.1, incubation_mean=5):
        super().__init__()
        self.N = N
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)

        self.beta = beta
        self.gamma = gamma
        self.steps = 0
        self.running = True

        # Gestione Lockdown
        self.social_distancing = 0.1  # Iniziale (basso distanziamento)
        self.lockdown_active = False

        # Creazione Agenti
        for _ in range(N):
            a = VirusAgent(self, beta, gamma, incubation_mean)
            self.agents.add(a)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(a, (x, y))

        # Paziente Zero
        agents_list = list(self.agents)
        if agents_list:
            patient_zero = self.random.choice(agents_list)
            patient_zero.state = "Exposed"
            patient_zero.infection_time = 0

        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: m.count_state("Susceptible"),
                "Exposed": lambda m: m.count_state("Exposed"),
                "Infected": lambda m: m.count_state("Infected"),
                "Recovered": lambda m: m.count_state("Recovered"),
            }
        )

    def step(self):
        # --- LOGICA LOCKDOWN DINAMICO ---
        # Se gli infetti superano il 10% della popolazione totale...
        infetti = self.count_state("Infected")
        if not self.lockdown_active and infetti > (self.N * 0.30):
            self.lockdown_active = True
            self.social_distancing = 0.90  # ...blocchiamo il 90% dei movimenti
            print(f"!!! STEP {self.steps}: LOCKDOWN ATTIVATO (Infetti > 30%) !!!")

        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        self.steps += 1

    def count_state(self, state):
        return sum(1 for a in self.agents if a.state == state)


# =========================
# 4. FUNZIONI DI VISUALIZZAZIONE
# =========================

def draw_grid(model, ax):
    """Disegna la mappa degli agenti"""
    grid_data = np.zeros((model.grid.width, model.grid.height))
    for a in model.agents:
        x, y = a.pos
        grid_data[x, y] = STATE_TO_INT[a.state]

    ax.clear()
    ax.imshow(grid_data, cmap=GRID_CMAP, vmin=0, vmax=4, interpolation='nearest')

    # Titolo dinamico
    status = "LOCKDOWN ATTIVO" if model.lockdown_active else "Libero Movimento"
    col = "red" if model.lockdown_active else "green"
    ax.set_title(f"Mappa ({status})", color=col, fontweight="bold")

    # Griglia estetica sottile
    ax.set_xticks(np.arange(-0.5, model.grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.grid.height, 1), minor=True)
    ax.grid(which='minor', color='#DDDDDD', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_curves_comparison(model, ax, ode_data):
    """Disegna il confronto Agenti vs ODE con legenda chiara"""
    df = model.datacollector.get_model_vars_dataframe()
    t_ode, S_ode, E_ode, I_ode, R_ode = ode_data

    ax.clear()

    # 1. Disegna le linee degli AGENTI (Solide)
    for i, state in enumerate(STATE_LABELS):
        if not df.empty:
            ax.plot(df[state], label=f"{state} (Agenti)", color=PLOT_COLORS[i], linewidth=2, alpha=0.9)

    # 2. Disegna le linee della TEORIA ODE (Tratteggiate)
    # Mostriamo principalmente gli Infetti ODE per confronto pulito
    limit = min(len(df), len(t_ode))
    if limit > 0:
        # Disegno l'ODE degli INFETTI (la più importante per il confronto picchi)
        ax.plot(t_ode[:limit], I_ode[:limit], linestyle='--', color='darkred', linewidth=2,
                label="Infected (Teoria ODE)")

    ax.set_title("Confronto: Simulazione Agenti vs Teoria Matematica")
    ax.set_xlabel("Step Temporali")
    ax.set_ylabel("Numero Agenti")
    ax.grid(True, linestyle=':', alpha=0.6)

    # Marker visivo sul grafico quando scatta il lockdown
    if model.lockdown_active:
        ax.text(0.02, 0.95, "LOCKDOWN ON", transform=ax.transAxes,
                color="red", fontsize=10, fontweight="bold",
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))

    # Legenda intelligente: mostra solo se ci sono dati
    if model.steps > 0:
        ax.legend(loc="center right", fontsize='small', framealpha=0.9)


def draw_bars(model, ax):
    """Istogramma dei totali"""
    counts = [model.count_state(s) for s in STATE_LABELS]
    ax.clear()
    bars = ax.bar(STATE_LABELS, counts, color=PLOT_COLORS)
    ax.set_ylim(0, model.N)
    ax.set_title("Distribuzione Istantanea")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')


# =========================
# MAIN
# =========================
def main():
    # --- PARAMETRI ---
    N_AGENTS = 800
    WIDTH = 50
    HEIGHT = 50
    BETA = 0.4
    GAMMA = 0.05
    INCUBATION = 5
    STEPS_TOTALI = 250

    # --- 1. PRE-CALCOLO MATEMATICA (ODE) ---
    print("Calcolo curve teoriche ODE...")
    t = np.linspace(0, STEPS_TOTALI, STEPS_TOTALI)
    y0 = (N_AGENTS - 1, 1, 0, 0)  # S, E, I, R iniziali
    # Parametri per ODE (Sigma = 1/Incubazione)
    ret = odeint(seir_ode, y0, t, args=(N_AGENTS, BETA, 1.0 / INCUBATION, GAMMA))
    ode_data = (t, ret[:, 0], ret[:, 1], ret[:, 2], ret[:, 3])

    # --- 2. INIZIALIZZA MODELLO ---
    model = VirusModel(N=N_AGENTS, width=WIDTH, height=HEIGHT,
                       beta=BETA, gamma=GAMMA, incubation_mean=INCUBATION)

    # Setup Grafica Live
    plt.ion()
    fig_live, (ax_grid, ax_curve, ax_bar) = plt.subplots(1, 3, figsize=(18, 6))
    fig_live.suptitle("Progetto SEIR: Agenti vs Equazioni Differenziali", fontsize=16)

    print("Avvio simulazione... (Premi Ctrl+C per interrompere e salvare)")

    try:
        for i in range(STEPS_TOTALI):
            if not plt.fignum_exists(fig_live.number):
                print("Finestra chiusa dall'utente.")
                break

            model.step()

            # Aggiornamento grafici live
            draw_grid(model, ax_grid)
            draw_curves_comparison(model, ax_curve, ode_data)
            draw_bars(model, ax_bar)

            plt.tight_layout()
            plt.pause(0.001)  # Pausa minima per fluidità

    except KeyboardInterrupt:
        print("\nInterruzione manuale richiesta.")

    finally:
        # --- 3. SALVATAGGIO FINALE 3 IMMAGINI ---
        plt.ioff()
        print("\nGenerazione immagini ad alta qualità in corso...")
        cwd = os.getcwd()

        # A. MAPPA (Quadrata)
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        draw_grid(model, ax1)
        path1 = os.path.join(cwd, "1_Mappa_Finale.png")
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # B. CURVE (Rettangolare Larga)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        draw_curves_comparison(model, ax2, ode_data)
        path2 = os.path.join(cwd, "2_Confronto_Curve.png")
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # C. BARRE (Verticale)
        fig3, ax3 = plt.subplots(figsize=(6, 8))
        draw_bars(model, ax3)
        path3 = os.path.join(cwd, "3_Statistiche_Finali.png")
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)

        print("=" * 50)
        print("✅ SALVATAGGIO COMPLETATO:")
        print(f"I file sono stati salvati in: {cwd}")
        print("1. 1_Mappa_Finale.png")
        print("2. 2_Confronto_Curve.png")
        print("3. 3_Statistiche_Finali.png")
        print("=" * 50)

        plt.show()


if __name__ == "__main__":
    main()

