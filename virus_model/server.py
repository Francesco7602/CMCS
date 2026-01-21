import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
from mesa import Agent, Model
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector

# =========================
# CONFIGURAZIONE
# =========================
PLOT_COLORS = ["tab:blue", "gold", "tab:red", "tab:gray"]
STATE_LABELS = ["Susceptible", "Exposed", "Infected", "Recovered"]
STATE_TO_INT = {"Susceptible": 1, "Exposed": 2, "Infected": 3, "Recovered": 4}
GRID_CMAP = ListedColormap(["white"] + PLOT_COLORS)


# =========================
# 1. MODULO DATI & FITTING
# =========================
def get_real_data_mock():
    t_data = np.linspace(0, 50, 50)
    I_data = 800 * np.exp(-(t_data - 25) ** 2 / 100) + np.random.normal(0, 10, 50)
    I_data = np.clip(I_data, 0, 800)
    return t_data, I_data


def seir_ode_fit(t, beta, gamma, N, sigma=1.0 / 5.0):
    def system(y, t):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    y0 = (N - 1, 1, 0, 0)
    ret = odeint(system, y0, t)
    return ret[:, 2]


def fit_parameters(N):
    print("--- FASE 1: CALIBRAZIONE PARAMETRI ---")
    t_real, I_real = get_real_data_mock()
    try:
        popt, _ = curve_fit(lambda t, b, g: seir_ode_fit(t, b, g, N),
                            t_real, I_real, p0=[0.5, 0.1], bounds=([0, 0], [3, 1]))
        return popt[0], popt[1]
    except:
        return 0.4, 0.05


def seir_ode_full(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# =========================
# 2. AGENTE
# =========================
class VirusAgent(Agent):
    def __init__(self, model, beta, gamma, incubation_mean):
        super().__init__(model)
        self.state = "Susceptible"
        self.beta = beta
        self.gamma = gamma
        self.infection_time = None
        self.incubation_period = max(1, int(self.random.gauss(incubation_mean, 1)))

    def step(self):
        # Movimento solo su Griglia
        if isinstance(self.model.grid, MultiGrid):
            if self.random.random() > self.model.social_distancing:
                self.move()

        self.progress_state()
        if self.state == "Infected":
            self.infect_neighbors()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if possible_steps:
            self.model.grid.move_agent(self, self.random.choice(possible_steps))

    def infect_neighbors(self):
        if isinstance(self.model.grid, NetworkGrid):
            # FIX: Otteniamo ID dei nodi e poi gli Agenti
            neighbor_nodes = list(self.model.grid.G.neighbors(self.pos))
            neighbors = self.model.grid.get_cell_list_contents(neighbor_nodes)
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        for n in neighbors:
            if n.state == "Susceptible" and self.random.random() < self.beta:
                n.state = "Exposed"
                n.infection_time = self.model.steps

    def progress_state(self):
        if self.state == "Exposed" and self.infection_time is not None:
            if self.model.steps - self.infection_time >= self.incubation_period:
                self.state = "Infected"
        elif self.state == "Infected" and self.random.random() < self.gamma:
            self.state = "Recovered"


# =========================
# 3. MODELLO
# =========================
class VirusModel(Model):
    def __init__(self, N=800, width=50, height=50, beta=0.5, gamma=0.1,
                 incubation_mean=5, topology="grid"):
        super().__init__()
        self.N = N
        self.width = width
        self.height = height
        self.topology = topology
        self.running = True
        self.steps = 0

        # Setup Topologia
        if topology == "network":
            self.G = nx.barabasi_albert_graph(N, 3)
            self.grid = NetworkGrid(self.G)
        else:
            self.grid = MultiGrid(width, height, torus=True)

        # Controllo
        self.social_distancing = 0.0
        self.lockdown_active = False
        self.thresh_high = 0.30
        self.thresh_low = 0.10

        # Creazione Agenti
        for i in range(N):
            a = VirusAgent(self, beta, gamma, incubation_mean)
            self.agents.add(a)
            if topology == "network":
                self.grid.place_agent(a, i)
            else:
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(a, (x, y))

        # Paziente Zero
        patient_zero = self.random.choice(list(self.agents))
        patient_zero.state = "Exposed"
        patient_zero.infection_time = 0

        # Datacollector Completo per i grafici finali
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: m.count_state("Susceptible"),
                "Exposed": lambda m: m.count_state("Exposed"),
                "Infected": lambda m: m.count_state("Infected"),
                "Recovered": lambda m: m.count_state("Recovered"),
            }
        )

    def step(self):
        infetti = self.count_state("Infected")
        pct = infetti / self.N

        if not self.lockdown_active and pct > self.thresh_high:
            self.lockdown_active = True
            self.social_distancing = 0.90
        elif self.lockdown_active and pct < self.thresh_low:
            self.lockdown_active = False
            self.social_distancing = 0.0

        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        self.steps += 1

    def count_state(self, state):
        return sum(1 for a in self.agents if a.state == state)


# =========================
# 4. REPORTING FINALE (NUOVA SEZIONE)
# =========================
def save_final_report(model, layout, ode_data):
    """Genera e salva le 3 immagini statiche richieste"""
    print("\nGenerazione Report Finale in corso...")
    cwd = os.getcwd()
    df = model.datacollector.get_model_vars_dataframe()

    # --- 1. Mappa Finale ---
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    colors = [PLOT_COLORS[STATE_TO_INT[a.state] - 1] for a in model.agents]
    if model.topology == "network":
        nx.draw(model.G, pos=layout, ax=ax1, node_size=50, node_color=colors,
                with_labels=False, width=0.5, edge_color="#CCCCCC")
        ax1.set_title("Stato Finale Rete")
    else:
        grid = np.zeros((model.grid.width, model.grid.height))
        for a in model.agents:
            grid[a.pos] = STATE_TO_INT[a.state]
        ax1.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=4, interpolation='nearest')
        ax1.set_title("Stato Finale Griglia")
    ax1.axis('off')
    fig1.savefig(os.path.join(cwd, "1_Mappa_Finale.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- 2. Curve Dettagliate (S, E, I, R vs ODE) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Curve Agenti
    for i, state in enumerate(STATE_LABELS):
        ax2.plot(df[state], label=f"{state} (Agenti)", color=PLOT_COLORS[i], linewidth=2)

    # Curva ODE (Solo Infetti per pulizia, o aggiungi le altre se vuoi)
    t_ode, I_ode = ode_data
    ax2.plot(t_ode, I_ode, '--', color="darkred", linewidth=2.5, label="Infected (ODE)")

    ax2.set_title("Confronto Completo: Agenti vs ODE")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Popolazione")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if model.lockdown_active:
        ax2.text(0.02, 0.95, "LOCKDOWN END STATE: ON", transform=ax2.transAxes, color="red")

    fig2.savefig(os.path.join(cwd, "2_Curve_Totali.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- 3. Istogramma Finale ---
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    counts = [model.count_state(s) for s in STATE_LABELS]
    bars = ax3.bar(STATE_LABELS, counts, color=PLOT_COLORS)
    ax3.set_title("Distribuzione Finale Popolazione")
    ax3.set_ylim(0, model.N * 1.1)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    fig3.savefig(os.path.join(cwd, "3_Istogramma.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print("✅ Salvataggio completato! File generati:")
    print("- 1_Mappa_Finale.png")
    print("- 2_Curve_Totali.png")
    print("- 3_Istogramma.png")


# =========================
# 5. GRAFICA LIVE & MAIN
# =========================
def run_batch_analysis(params, runs=10, steps=100):
    print(f"--- FASE 2: ANALISI BATCH ({runs} run) ---")
    all_infected = []
    for _ in range(runs):
        model = VirusModel(**params)
        history = []
        for _ in range(steps):
            model.step()
            history.append(model.count_state("Infected"))
        all_infected.append(history)
    data = np.array(all_infected)
    return np.mean(data, axis=0), np.std(data, axis=0)


def draw_viz_live(model, layout, ax_map, ax_curve, ode_data, batch_stats):
    """Versione leggera per l'animazione live"""
    ax_map.clear()
    colors = [PLOT_COLORS[STATE_TO_INT[a.state] - 1] for a in model.agents]

    if model.topology == "network":
        nx.draw(model.G, pos=layout, ax=ax_map, node_size=30, node_color=colors,
                with_labels=False, width=0.3, edge_color="#DDDDDD")
        ax_map.set_title(f"Rete (Lockdown: {model.lockdown_active})")
    else:
        grid = np.zeros((model.grid.width, model.grid.height))
        for a in model.agents:
            grid[a.pos] = STATE_TO_INT[a.state]
        ax_map.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=4, interpolation='nearest')
        ax_map.set_title(f"Griglia (Lockdown: {model.lockdown_active})")
    ax_map.axis('off')

    ax_curve.clear()
    mu, sigma = batch_stats
    t = np.arange(len(mu))
    ax_curve.fill_between(t, mu - sigma, mu + sigma, color='gray', alpha=0.2)

    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        ax_curve.plot(df["Infected"], color="red", linewidth=2, label="Live Infected")

    ax_curve.plot(ode_data[0], ode_data[1], '--', color="black", label="ODE Infected")
    ax_curve.legend(loc="upper right", fontsize='small')
    ax_curve.set_title("Andamento Live")


def main():
    N = 500
    STEPS = 120

    print("Scegli Topologia: [1] Griglia, [2] Network")
    choice = input("> ")
    topo = "network" if choice == "2" else "grid"

    beta, gamma = fit_parameters(N)

    params = {
        "N": N, "width": 30, "height": 30,
        "beta": beta, "gamma": gamma,
        "incubation_mean": 5, "topology": topo
    }

    mu_batch, std_batch = run_batch_analysis(params, runs=10, steps=STEPS)

    t_ode = np.linspace(0, STEPS, STEPS)
    ret = odeint(seir_ode_full, (N - 1, 1, 0, 0), t_ode, args=(N, beta, 1 / 5, gamma))
    ode_data = (t_ode, ret[:, 2])

    print("--- FASE 3: LIVE SIMULATION ---")
    model = VirusModel(**params)

    layout = None
    if topo == "network":
        print("Calcolo layout grafico...")
        layout = nx.spring_layout(model.G, seed=42, iterations=50)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    try:
        for i in range(STEPS):
            if not plt.fignum_exists(fig.number): break
            model.step()
            draw_viz_live(model, layout, ax1, ax2, ode_data, (mu_batch, std_batch))
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nInterruzione manuale.")

    plt.ioff()
    plt.close()  # Chiude la finestra live per non fare confusione

    # --- FASE 4: SALVATAGGIO REPORT ---
    save_final_report(model, layout, ode_data)

    print("Premi Enter per chiudere...")
    # input()


if __name__ == "__main__":
    main()