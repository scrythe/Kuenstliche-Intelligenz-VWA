from matplotlib import animation
import matplotlib.pyplot as plt

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src import sin
import numpy as np


eingaben = np.arange(0, 5, 0.1)
eingaben = eingaben.reshape(len(eingaben), 1)
ziele = np.sin(eingaben)

_, geschichte = sin.trainiere_netzwerk(eingaben,ziele)

fig, ax = plt.subplots(figsize=(10, 5))

plt.plot(eingaben, ziele, label="Wahre Funktion")
(linie,) = ax.plot([], [], label="Neuronales Netzwerk", color="orange")

framge_range = range(0, len(geschichte), 10)

def init():
    linie.set_xdata(eingaben)
    return (linie,)

def update(epoche):
    linie.set_ydata(geschichte[epoche])
    ax.set_title(f"Epoche {epoche}")
    return (linie,)

ani = animation.FuncAnimation(
    fig, update, frames=framge_range, init_func=init, interval=100, repeat=False
)

plt.legend()



plt.show()
