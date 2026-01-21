# virus_model/refactored_model/plotting.py

import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import networkx as nx

from .constants import (
    OUTPUT_DIR,
    AGENT_COLORS,
    SHORT_LABELS,
    STATE_EMPTY,
    GRID_CMAP
)

def save_single_run_results(model, df, ode_data):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # --- CURVE ---
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)

    if ode_data is not None:
        ax1.plot(ode_data["t"], ode_data["S"], '--', color=AGENT_COLORS[0], alpha=0.4, label="S (ODE)")
        ax1.plot(ode_data["t"], ode_data["E"], '--', color=AGENT_COLORS[1], alpha=0.4, label="E (ODE)")
        ax1.plot(ode_data["t"], ode_data["I"], '--', color=AGENT_COLORS[3], alpha=0.4, label="I (ODE)")

    if not df.empty:
        ax1.plot(df["S"], label="Susceptible", color=AGENT_COLORS[0])
        ax1.plot(df["E"], label="Exposed", color=AGENT_COLORS[1])
        ax1.plot(df["I_asymp"], label="Hidden", color=AGENT_COLORS[2], linestyle="-.")
        ax1.plot(df["I_symp"], label="Detected", color=AGENT_COLORS[3])
        ax1.plot(df["R"], label="Recovered", color=AGENT_COLORS[4])

        if "Lockdown" in df.columns:
            lockdown_steps = df[df["Lockdown"] == 1].index
            if len(lockdown_steps) > 0:
                ax1.axvspan(lockdown_steps[0], lockdown_steps[-1], color='red', alpha=0.1, label="Lockdown")

    ax1.set_title(f"Report Run (Simulated vs ODE) - {timestamp}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    path_curves = os.path.join(OUTPUT_DIR, f"run_{timestamp}_curves.png")
    fig1.savefig(path_curves, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- MAPPA / GRIGLIA ---
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111)

    if model.topology == "network":
        colors = [AGENT_COLORS[a.state] for a in model.agents]
        pos = nx.spring_layout(model.G, seed=42)
        nx.draw(model.G, pos=pos, ax=ax2, node_size=50, node_color=colors, width=0.5, edge_color="#CCCCCC")
        ax2.set_title(f"Stato Finale Rete - {timestamp}")
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(AGENT_COLORS, SHORT_LABELS)]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize='x-small')
    else:
        # Inizializziamo con STATE_EMPTY (-1) invece di 0!
        grid_arr = np.full((model.grid.width, model.grid.height), STATE_EMPTY)
        for a in model.agents:
            grid_arr[a.pos] = a.state

        # imshow con vmin=-1 (Bianco) e vmax=4 (Grigio)
        ax2.imshow(grid_arr, cmap=GRID_CMAP, vmin=-1, vmax=4, interpolation="nearest")
        ax2.set_title(f"Stato Finale Griglia - {timestamp}")

        legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(AGENT_COLORS, SHORT_LABELS)]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')

    ax2.axis('off')
    path_net = os.path.join(OUTPUT_DIR, f"run_{timestamp}_map.png")
    fig2.savefig(path_net, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    return path_curves, path_net


def save_batch_results_plot(peaks):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.hist(peaks, bins=15, color="purple", alpha=0.7, edgecolor='black')
    ax3.set_title(f"Analisi Stocastica Batch ({len(peaks)} runs) - {timestamp}")
    ax3.set_xlabel("Picco Massimo Infetti")
    ax3.set_ylabel("Frequenza")
    ax3.grid(axis='y', alpha=0.5, linestyle='--')
    path_batch = os.path.join(OUTPUT_DIR, f"batch_{timestamp}_histogram.png")
    fig3.savefig(path_batch, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    return path_batch
