# virus_model/refactored_model/constants.py
"""
This file defines global constants used throughout the simulation model.
It includes definitions for age groups, agent states, color schemes for plotting,
and configuration for output directories.
"""

import os
from matplotlib.colors import ListedColormap
import numpy as np


# Age Group Indices
CHILD = 0
TEEN = 1
ADULT = 2
SENIOR = 3

# Labels for age groups
AGE_LABELS = ["Child", "Teen", "Adult", "Senior"]

# Case Fatality Rate per age group
MU_GROUP = [0.00009, 0.00005, 0.00688, 0.15987]

# Agent state constants
STATE_EMPTY = -1  # Represents an empty cell in the grid
STATE_SUSCEPTIBLE = 0
STATE_EXPOSED = 1
STATE_INFECTED_ASYMPTOMATIC = 2
STATE_INFECTED_SYMPTOMATIC = 3
STATE_RECOVERED = 4

# Short labels for agent states, used in plots and charts
SHORT_LABELS = ["S", "E", "I_hid", "I_det", "R"]

# Color definitions for visualization.
# The order corresponds to agent states: Empty, Susceptible, Exposed, Asymptomatic, Symptomatic, Recovered.
# Map order: [White, Blue, Gold, Orange, Red, Gray]
# Associated indices: [-1, 0, 1, 2, 3, 4]
PLOT_COLORS = ["white", "tab:blue", "gold", "tab:orange", "tab:red", "tab:gray"]

# Matplotlib Colormap for grid visualization (includes the color for empty cells)
GRID_CMAP = ListedColormap(PLOT_COLORS)

# Colors for agent-based plots (e.g., line charts), excluding the 'empty' state color
AGENT_COLORS = PLOT_COLORS[1:]

# Directory for saving simulation output data and plots
OUTPUT_DIR = "output_data"
# Create the output directory if it does not already exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
