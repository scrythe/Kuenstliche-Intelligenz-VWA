import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from neuronales_netzwerk.verlustfunktionen import MittlererQuadratischerFehler


class Netzwerk:
    def __init__(self):
        self.gewicht = 0.1
        self.bias = 0.1
        self.verlustfunktion = MittlererQuadratischerFehler

    def vorwaerts_durchlauf(self, eingaben):
        return self.gewicht * eingaben + self.bias

    def rueckwaerts_durchlauf(self, eingaben, vorhersagen, ziele):
        # dL/dA
        verlust_gradient = self.verlustfunktion.rueckwaerts(vorhersagen, ziele)
        # dL/dW
        gewicht_gradient = np.dot(eingaben.T, verlust_gradient)
        # dL/db
        bias_gradient = np.sum(verlust_gradient)
        return (gewicht_gradient, bias_gradient)

    def gradientenabstieg(self, eingaben, ziele):
        lernrate = 0.01
        geschichte = []
        for _ in range(250):
            vorhersagen = self.vorwaerts_durchlauf(eingaben)
            geschichte.append(self.vorwaerts_durchlauf(eingaben))
            gewicht_gradient, bias_gradient = self.rueckwaerts_durchlauf(
                eingaben, vorhersagen, ziele
            )
            self.gewicht -= lernrate * gewicht_gradient
            self.bias -= lernrate * bias_gradient
        return geschichte


def f(x):
    return 4 * x + 3


eingaben = np.arange(-5, 5 - 0.1)
ziele = f(eingaben)

netzwerk = Netzwerk()
geschichte = netzwerk.gradientenabstieg(eingaben, ziele)
vorhersagen = netzwerk.vorwaerts_durchlauf(eingaben)


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
