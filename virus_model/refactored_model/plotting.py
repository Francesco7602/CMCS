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

def save_single_run_results(model, df, ode_data, gillespie_data):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # --- CURVE ---
    fig1 = plt.figure(figsize=(12, 4))
    ax1 = fig1.add_subplot(111)

    if ode_data is not None:
        ax1.plot(ode_data["t"], ode_data["S"], '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
        ax1.plot(ode_data["t"], ode_data["E"], '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
        ax1.plot(ode_data["t"], ode_data["I"], '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
        ax1.plot(ode_data["t"], ode_data["R"], '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")

    if gillespie_data is not None:
        ax1.plot(gillespie_data["time"], gillespie_data["S"], ':', color=AGENT_COLORS[0], alpha=0.9, label="S (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["E"], ':', color=AGENT_COLORS[1], alpha=0.9, label="E (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["I"], ':', color=AGENT_COLORS[3], alpha=0.9, label="I (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["R"], ':', color=AGENT_COLORS[4], alpha=0.9, label="R (Gillespie)")

    if not df.empty:
        ax1.plot(df.index, df["S"], label="S (ABM)", color=AGENT_COLORS[0])
        ax1.plot(df.index, df["E"], label="E (ABM)", color=AGENT_COLORS[1])
        ax1.plot(df.index, df["I_asymp"] + df["I_symp"], label="I (ABM)", color=AGENT_COLORS[3])
        ax1.plot(df.index, df["R"], label="R (ABM)", color=AGENT_COLORS[4])

        if "Lockdown" in df.columns:
            lockdown_steps = df[df["Lockdown"] == 1].index
            if len(lockdown_steps) > 0:
                ax1.axvspan(lockdown_steps[0], lockdown_steps[-1], color='red', alpha=0.1, label="Lockdown")

    ax1.set_title(f"Run Report (ABM vs ODE vs Gillespie) - {timestamp}")
    ax1.set_xlim(0, max(df.index) if not df.empty else 100)
    ax1.set_ylim(0, model.N)
    ax1.legend(loc="upper right", fontsize='x-small', ncol=3)
    ax1.grid(True, alpha=0.3)

    path_curves = os.path.join(OUTPUT_DIR, f"run_{timestamp}_curves.png")
    fig1.savefig(path_curves, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- MAPPA / GRIGLIA ---
    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111)

    if model.topology == "network":
        colors = [AGENT_COLORS[a.state] for a in model.agents]
        pos = nx.spring_layout(model.G, seed=42)
        nx.draw(model.G, pos=pos, ax=ax2, node_size=50, node_color=colors, width=0.5, edge_color="#CCCCCC")
        ax2.set_title(f"Stato Finale Rete - {timestamp}")
    else:
        grid_arr = np.full((model.grid.width, model.grid.height), STATE_EMPTY)
        for a in model.agents:
            grid_arr[a.pos] = a.state
        ax2.imshow(grid_arr, cmap=GRID_CMAP, vmin=-1, vmax=4, interpolation="nearest")
        ax2.set_title(f"Stato Finale Griglia - {timestamp}")
        
    legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(AGENT_COLORS, SHORT_LABELS)]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='x-small')
    ax2.axis('off')
    path_map = os.path.join(OUTPUT_DIR, f"run_{timestamp}_map.png")
    fig2.savefig(path_map, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # --- RETE DI PETRI ---
    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111)
    if not df.empty:
        latest = df.iloc[-1]
        S, E, I, R = latest["S"], latest["E"], latest["I_asymp"] + latest["I_symp"], latest["R"]
        draw_petri_net(ax3, S, E, I, R)
        ax3.set_title(f"Rete di Petri (Stato Finale) - {timestamp}")
    
    path_petri = os.path.join(OUTPUT_DIR, f"run_{timestamp}_petri.png")
    fig3.savefig(path_petri, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return path_curves, path_map, path_petri


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


def draw_petri_net(ax, S, E, I, R):
    ax.clear()
    ax.set_title("Rete di Petri (Flusso Agenti)")
    
    # Posizioni fisse per posti e transizioni
    places = {
        'S': {'pos': (0.1, 0.5), 'count': S, 'color': AGENT_COLORS[0]},
        'E': {'pos': (0.4, 0.8), 'count': E, 'color': AGENT_COLORS[1]},
        'I': {'pos': (0.7, 0.5), 'count': I, 'color': AGENT_COLORS[3]},
        'R': {'pos': (0.4, 0.2), 'count': R, 'color': AGENT_COLORS[4]}
    }
    transitions = {
        'infettare': {'pos': (0.25, 0.65), 'label': 'β'},
        'incubare': {'pos': (0.55, 0.65), 'label': 'σ'},
        'guarire': {'pos': (0.55, 0.35), 'label': 'γ'}
    }
    
    # Disegna posti (cerchi)
    for name, p in places.items():
        circle = plt.Circle(p['pos'], radius=0.1, facecolor=p['color'], edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(p['pos'][0], p['pos'][1], f"{p['count']}", ha='center', va='center', fontsize=12, color='white', weight='bold')
        ax.text(p['pos'][0], p['pos'][1] - 0.15, name, ha='center', va='center', fontsize=12)

    # Disegna transizioni (rettangoli)
    for name, t in transitions.items():
        rect = plt.Rectangle((t['pos'][0] - 0.05, t['pos'][1] - 0.05), 0.1, 0.1, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)
        ax.text(t['pos'][0], t['pos'][1], t['label'], ha='center', va='center', fontsize=14, color='white')

    # Disegna archi (frecce)
    # S -> infettare (beta)
    ax.arrow(0.2, 0.5, 0.05, 0.1, head_width=0.02, head_length=0.02, fc='k', ec='k') 
    # infettare (beta) -> E
    ax.arrow(0.25, 0.7, 0.1, 0.05, head_width=0.02, head_length=0.02, fc='k', ec='k') 
    # E -> incubare (sigma)
    ax.arrow(0.47, 0.73, 0.03, -0.03, head_width=0.02, head_length=0.02, fc='k', ec='k') 
    # incubare (sigma) -> I
    ax.arrow(0.6, 0.65, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k') 
    # I -> guarire (gamma)
    ax.arrow(0.63, 0.43, -0.03, -0.05, head_width=0.02, head_length=0.02, fc='k', ec='k') 
    # guarire (gamma) -> R
    ax.arrow(0.5, 0.35, -0.05, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
