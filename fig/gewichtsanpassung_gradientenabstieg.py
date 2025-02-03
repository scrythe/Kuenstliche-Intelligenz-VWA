import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from neuronales_netzwerk.verlustfunktionen import MittlererQuadratischerFehler


def f(x):
    return 2 * x


eingaben = np.arange(0, 4, 0.1)
ziele = f(eingaben)


def berechne_kostenfunktion_auf_gewicht(gewichte):
    kosten_gewicht = []
    for gewicht in gewichte:
        vorhersagen = gewicht * eingaben
        kosten = MittlererQuadratischerFehler.kosten(vorhersagen, ziele)
        kosten_gewicht.append(kosten)
    return kosten_gewicht


def berechne_gradient(gewicht):
    vorhersagen = gewicht * eingaben
    verlust_gradient = MittlererQuadratischerFehler.rueckwaerts(vorhersagen, ziele)

    gradient_gewicht = np.dot(eingaben.T, verlust_gradient)
    return [gradient_gewicht]


def zeichne_hinunterrollen():
    geschichte_gewichte = []
    geschichte_kosten = []
    gewicht = 0  # Start Gewicht
    for _ in range(30):
        gradient = berechne_gradient(gewicht)
        gewicht -= 0.01 * gradient[0]
        geschichte_gewichte.append(gewicht)
        vorhersagen = gewicht * eingaben
        kosten = MittlererQuadratischerFehler.kosten(vorhersagen, ziele)
        geschichte_kosten.append(kosten)
    return (geschichte_gewichte, geschichte_kosten)


gewichte = np.arange(0, 4, 0.1)
kosten_gewicht = berechne_kostenfunktion_auf_gewicht(gewichte)
geschichte_gewichte, geschichte_kosten = zeichne_hinunterrollen()

# plt.plot(eingaben, ziele)
plt.plot(eingaben, kosten_gewicht)

plt.scatter(geschichte_gewichte, geschichte_kosten, color="red", zorder=2)
plt.plot(geschichte_gewichte, geschichte_kosten, color="red", linestyle="--")

plt.xlabel("Gewichte (w)")
plt.ylabel("Kosten (L)")

# plt.scatter()
plt.show()
