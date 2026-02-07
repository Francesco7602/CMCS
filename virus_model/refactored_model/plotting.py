# virus_model/refactored_model/plotting.py
"""
This file contains utility functions for generating and saving plots of the
simulation results. It uses Matplotlib to create visualizations for single runs,
batch analyses, and parameter sweeps.
"""

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
    """
    Generates and saves a comprehensive report for a single simulation run.

    The report includes three plots:
    1.  Epidemic curves (ABM vs. ODE vs. Gillespie).
    2.  The final spatial state of the model (grid or network).
    3.  A Transition Schema diagram representing the final distribution of agents.

    Args:
        model (VirusModel): The completed model instance.
        df (pd.DataFrame): The data collected from the ABM run.
        ode_data (dict): The data from the ODE simulation.
        gillespie_data (pd.DataFrame): The data from the Gillespie simulation.

    Returns:
        tuple: A tuple of file paths for the saved plots (curves, map, petri).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # --- 1. Epidemic Curves Plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    # Plot ODE results (dashed lines)
    if ode_data is not None:
        ax1.plot(ode_data["t"], ode_data["S"] / model.N * 100, '--', color=AGENT_COLORS[0], alpha=0.5, label="S (ODE)")
        ax1.plot(ode_data["t"], ode_data["E"] / model.N * 100, '--', color=AGENT_COLORS[1], alpha=0.5, label="E (ODE)")
        ax1.plot(ode_data["t"], ode_data["I"] / model.N * 100, '--', color=AGENT_COLORS[3], alpha=0.5, label="I (ODE)")
        ax1.plot(ode_data["t"], ode_data["R"] / model.N * 100, '--', color=AGENT_COLORS[4], alpha=0.5, label="R (ODE)")

    # Plot Gillespie results (dotted lines)
    if gillespie_data is not None:
        ax1.plot(gillespie_data["time"], gillespie_data["S"] / model.N * 100, ':', color=AGENT_COLORS[0], alpha=0.9,
                 label="S (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["E"] / model.N * 100, ':', color=AGENT_COLORS[1], alpha=0.9,
                 label="E (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["I"] / model.N * 100, ':', color=AGENT_COLORS[3], alpha=0.9,
                 label="I (Gillespie)")
        ax1.plot(gillespie_data["time"], gillespie_data["R"] / model.N * 100, ':', color=AGENT_COLORS[4], alpha=0.9,
                 label="R (Gillespie)")

    # Plot ABM results (solid lines)
    if not df.empty:
        ax1.plot(df.index, df["S"] / model.N * 100, label="S (ABM)", color=AGENT_COLORS[0])
        ax1.plot(df.index, df["E"] / model.N * 100, label="E (ABM)", color=AGENT_COLORS[1])
        ax1.plot(df.index, (df["I_asymp"] + df["I_symp"]) / model.N * 100, label="I (ABM)", color=AGENT_COLORS[3])
        ax1.plot(df.index, df["R"] / model.N * 100, label="R (ABM)", color=AGENT_COLORS[4])

        if "Lockdown" in df.columns:
            lockdown_steps = df[df["Lockdown"] == 1].index
            if len(lockdown_steps) > 0:
                ax1.axvspan(lockdown_steps[0], lockdown_steps[-1], color='red', alpha=0.1, label="Lockdown")

    ax1.set_title("Run Report (ABM vs ODE vs Gillespie)")
    ax1.set_xlim(0, max(df.index) if not df.empty else 100)
    ax1.set_ylabel("Population (%) - Log Scale")
    ax1.legend(loc="upper right", fontsize='x-small', ncol=2)
    ax1.grid(True, which="both", alpha=0.3)

    path_curves = os.path.join(OUTPUT_DIR, f"run_{timestamp}_curves.png")
    fig1.savefig(path_curves, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- 2. Final State Map/Grid Plot ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    if model.G is not None: # Network topology
        colors = [AGENT_COLORS[a.state] for a in model.agents]
        pos = model.layout if model.layout is not None else nx.spring_layout(model.G, seed=42)
        nx.draw(model.G, pos=pos, ax=ax2, node_size=50, node_color=colors, width=0.5, edge_color="#CCCCCC")
        ax2.set_title(f"Final Network State - {timestamp}")
    else: # Grid topology
        grid_arr = np.full((model.grid.width, model.grid.height), STATE_EMPTY)
        for a in model.agents:
            grid_arr[a.pos] = a.state
        ax2.imshow(grid_arr, cmap=GRID_CMAP, vmin=-1, vmax=4, interpolation="nearest")
        ax2.set_title(f"Final Grid State - {timestamp}")
        
    legend_elements = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(AGENT_COLORS, SHORT_LABELS)]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='x-small')
    ax2.axis('off')
    path_map = os.path.join(OUTPUT_DIR, f"run_{timestamp}_map.png")
    fig2.savefig(path_map, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # --- 3. Transition Schema Plot ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    if not df.empty:
        latest = df.iloc[-1]
        S, E, I_asymp, I_symp, R = latest["S"], latest["E"], latest["I_asymp"], latest["I_symp"], latest["R"]
        draw_petri_net(ax3, S, E, I_asymp, I_symp, R)

    path_petri = os.path.join(OUTPUT_DIR, f"run_{timestamp}_petri.png")
    fig3.savefig(path_petri, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return path_curves, path_map, path_petri


def save_batch_results_plot(peaks, threshold=None):
    """
    Saves a histogram of the epidemic peaks from a batch run.

    Args:
        peaks (list): A list of the maximum number of infected individuals from each run.
        threshold (int, optional): A risk threshold to display as a vertical line. Defaults to None.

    Returns:
        str: The file path of the saved histogram plot.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of peaks
    ax.hist(peaks, bins=15, color="purple", alpha=0.7, edgecolor='black')

    # Threshold line
    if threshold is not None:
        ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f"Threshold ({threshold})")
        ax.legend()

    ax.set_title(f"Stochastic Batch Analysis ({len(peaks)} runs) - {timestamp}")
    ax.set_xlabel("Maximum Infected Peak")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.5, linestyle='--')

    path_batch = os.path.join(OUTPUT_DIR, f"batch_{timestamp}_histogram.png")
    fig.savefig(path_batch, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return path_batch


def draw_petri_net(ax, S, E, I_asymp, I_symp, R):
    """
    Draws a Transition Schema diagram on a given Matplotlib Axes object.

    The diagram visualizes the flow of agents between SEIR compartments.

    Args:
        ax (matplotlib.axes.Axes): The axes to draw on.
        S (int): Number of agents in the Susceptible compartment.
        E (int): Number of agents in the Exposed compartment.
        I_asymp (int): Number of asymptomatic Infected agents.
        I_symp (int): Number of symptomatic Infected agents.
        R (int): Number of agents in the Recovered compartment.
    """
    ax.clear()

    # Layout parameters
    x_space, y_space = 0.22, 0.35
    r_node = 0.06
    width_t, height_t = 0.14, 0.09
    x0, y0 = 0.08, 0.5

    # Define places (compartments) and transitions
    places = {
        'S': {'pos': (x0, y0), 'count': S, 'color': AGENT_COLORS[0]},
        'E': {'pos': (x0 + 2 * x_space, y0), 'count': E, 'color': AGENT_COLORS[1]},
        'I_asymp': {'pos': (x0 + 3.5 * x_space, y0 + y_space), 'count': I_asymp, 'color': AGENT_COLORS[2]},
        'I_symp': {'pos': (x0 + 3.5 * x_space, y0 - y_space), 'count': I_symp, 'color': AGENT_COLORS[3]},
        'R': {'pos': (x0 + 5 * x_space, y0), 'count': R, 'color': AGENT_COLORS[4]}
    }
    transitions = {
        'T1': {'pos': (x0 + x_space, y0), 'label': 'β'},
        'T2': {'pos': (x0 + 2.75 * x_space, y0 + 0.5 * y_space), 'label': 'σ(1-p)'},
        'T3': {'pos': (x0 + 2.75 * x_space, y0 - 0.5 * y_space), 'label': 'σ*p'},
        'T4': {'pos': (x0 + 4.25 * x_space, y0 + 0.5 * y_space), 'label': 'γ'},
        'T5': {'pos': (x0 + 4.25 * x_space, y0 - 0.5 * y_space), 'label': 'γ'}
    }
    edges = [
        (('p', 'S'), ('t', 'T1')), (('t', 'T1'), ('p', 'E')),
        (('p', 'E'), ('t', 'T2')), (('t', 'T2'), ('p', 'I_asymp')),
        (('p', 'E'), ('t', 'T3')), (('t', 'T3'), ('p', 'I_symp')),
        (('p', 'I_asymp'), ('t', 'T4')), (('t', 'T4'), ('p', 'R')),
        (('p', 'I_symp'), ('t', 'T5')), (('t', 'T5'), ('p', 'R'))
    ]

    # Draw transitions (rectangles)
    for t in transitions.values():
        rect = plt.Rectangle((t['pos'][0] - width_t / 2, t['pos'][1] - height_t / 2),
                             width_t, height_t, facecolor='gray', edgecolor='black', zorder=4)
        ax.add_patch(rect)
        ax.text(t['pos'][0], t['pos'][1], t['label'], ha='center', va='center',
                fontsize=9, color='white', weight='bold', zorder=5)

    # Draw places (circles) and their labels
    for name, p in places.items():
        circle = plt.Circle(p['pos'], radius=r_node, facecolor=p['color'], edgecolor='black', alpha=0.8, zorder=4)
        ax.add_patch(circle)
        ax.text(p['pos'][0], p['pos'][1], f"{int(p['count'])}", ha='center', va='center',
                fontsize=9, color='white', weight='bold', zorder=5)
        label_y_offset = 0.12 if name == 'I_asymp' else -0.12
        ax.text(p['pos'][0], p['pos'][1] + label_y_offset, name,
                ha='center', va='center', fontsize=10, weight='bold')

    # Draw edges (arrows) between places and transitions
    for start, end in edges:
        src_pos = places[start[1]]['pos'] if start[0] == 'p' else transitions[start[1]]['pos']
        dst_pos = places[end[1]]['pos'] if end[0] == 'p' else transitions[end[1]]['pos']
        ax.annotate("", xy=dst_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2,
                                    shrinkA=18, shrinkB=18, mutation_scale=12), zorder=3)

    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1)
    ax.axis('off')

def save_sweep_results_plot(results, param_name):
    """
    Saves a plot of the results from a parameter sweep.

    The plot shows how the average epidemic peak changes as the specified
    parameter is varied.

    Args:
        results (list): A list of tuples, where each tuple is `(parameter_value, avg_peak)`.
        param_name (str): The name of the parameter that was swept (e.g., 'beta').

    Returns:
        str: The file path of the saved plot.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(8, 4))

    # Unpack the results into x and y values
    x_val = [r[0] for r in results]
    y_val = [r[1] for r in results]

    ax.plot(x_val, y_val, '-o', color='teal', linewidth=2)

    ax.set_title(f"Parameter Sweep: {param_name} - {timestamp}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Average Infection Peak (Max)")
    ax.grid(True, linestyle='--', alpha=0.7)

    filename = f"sweep_{param_name}_{timestamp}.png"
    plot_path = os.path.join(OUTPUT_DIR, filename)

    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return plot_path