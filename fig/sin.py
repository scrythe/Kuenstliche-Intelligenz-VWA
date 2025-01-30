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

netzwerk,_ = sin.trainiere_netzwerk(eingaben,ziele)


vorhersagen = netzwerk.vorwaerts_durchlauf(eingaben)
plt.plot(eingaben, ziele, label="Wahre Funktion")
plt.plot(eingaben, vorhersagen, label="Neuronales Netzwerk", color="orange")

plt.legend()
plt.show()
