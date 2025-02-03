import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from neuronales_netzwerk.verlustfunktionen import MittlererQuadratischerFehler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


def f(x):
    return 2 * x + 2


eingaben = np.arange(-3, 3, 0.1)
ziele = f(eingaben)


def berechne_kostenfunktion_auf_gewicht_bias(gewichte, bias):
    vorhersagen = gewichte * eingaben + bias
    kosten = MittlererQuadratischerFehler.kosten(vorhersagen, ziele)
    return kosten


def berechne_gradient(gewicht, bias):
    vorhersagen = gewicht * eingaben + bias
    verlust_gradient = MittlererQuadratischerFehler.rueckwaerts(vorhersagen, ziele)

    gradient_gewicht = np.dot(eingaben.T, verlust_gradient)
    gradient_bias = np.sum(verlust_gradient)
    return [gradient_gewicht, gradient_bias]


def zeichne_landschaft():
    gewichtsbereich = np.arange(-1, 4, 0.1)
    biasbereich = np.arange(-1, 4, 0.1)
    gewichte, bias = np.meshgrid(gewichtsbereich, biasbereich)
    verluste = np.array(
        [
            [
                berechne_kostenfunktion_auf_gewicht_bias(gewicht, bias_wert)
                for gewicht in gewichtsbereich
            ]
            for bias_wert in biasbereich
        ]
    )

    ax.plot_surface(
        gewichte,
        bias,
        verluste,
        cmap="coolwarm",
    )  # Kostenlandschaft


def berechne_hinunterrollen():
    geschichte = []
    gewicht = -1  # Start Gewicht
    bias = -1  # Start Bias
    for _ in range(30):
        vorhersagen = gewicht * eingaben + bias
        kosten = MittlererQuadratischerFehler.kosten(vorhersagen, ziele)

        geschichte.append((gewicht, bias, kosten))

        gradient = berechne_gradient(gewicht, bias)
        gewicht -= 0.02 * gradient[0]
        bias -= 0.02 * gradient[1]

    return geschichte


def zeichne_hinunterrollen(punkte):
    gewichte = [p[0] for p in punkte]
    bias = [p[1] for p in punkte]
    kosten = [p[2] for p in punkte]
    ax.plot(
        gewichte,
        bias,
        kosten,
        color="red",
        marker="o",
        markersize=5,
        label="Descent Path",
    )
    for i in range(len(punkte) - 1):
        gewicht, bias, kosten = punkte[i]
        dgewicht = punkte[i + 1][0] - gewicht
        dbias = punkte[i + 1][1] - bias
        dkosten = punkte[i + 1][2] - kosten
        ax.quiver(
            gewicht,
            bias,
            kosten,
            dgewicht,
            dbias,
            dkosten,
            arrow_length_ratio=0.5,
            color="black",
        )


fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection="3d", computed_zorder=False)

zeichne_landschaft()

geschichte = berechne_hinunterrollen()
zeichne_hinunterrollen(geschichte)

# Achsentitel
ax.set_title("Kostenlandschaft in Bezug auf Gewichte und Bias")
ax.set_xlabel("Gewicht (w)")
ax.set_ylabel("Bias (b)")
ax.invert_yaxis()
ax.set_zlabel("Kosten (L)")

plt.show()
