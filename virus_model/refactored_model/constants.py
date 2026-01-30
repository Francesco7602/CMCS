# virus_model/refactored_model/constants.py

import os
from matplotlib.colors import ListedColormap
import numpy as np


# Age Group Indices
CHILD = 0
TEEN = 1
ADULT = 2
SENIOR = 3

AGE_LABELS = ["Child", "Teen", "Adult", "Senior"]

MU_GROUP = [0.00009, 0.00005, 0.00688, 0.15987] # Case Fatality Rate

STATE_EMPTY = -1
STATE_SUSCEPTIBLE = 0
STATE_EXPOSED = 1
STATE_INFECTED_ASYMPTOMATIC = 2
STATE_INFECTED_SYMPTOMATIC = 3
STATE_RECOVERED = 4

# Etichette
SHORT_LABELS = ["S", "E", "I_hid", "I_det", "R"]

# DEFINIZIONE COLORI: BIANCO all'inizio per lo stato -1 (Vuoto)
# Ordine mappa: [Bianco, Blu, Oro, Arancio, Rosso, Grigio]
# Indici associati: [-1, 0, 1, 2, 3, 4]
PLOT_COLORS = ["white", "tab:blue", "gold", "tab:orange", "tab:red", "tab:gray"]

# Colormap per la griglia (include il bianco)
GRID_CMAP = ListedColormap(PLOT_COLORS)

# Colori solo per gli agenti (per i grafici delle curve, escludendo il bianco)
AGENT_COLORS = PLOT_COLORS[1:]

OUTPUT_DIR = "output_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
