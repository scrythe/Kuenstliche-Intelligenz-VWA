import numpy as np
from .neuronales_netzwerk import Netzwerk
from .neuronale_schicht import Schicht
from .aktivierungsfunktionen import ReLU, Linear
from .verlustfunktionen import MittlererQuadratischerFehler
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import ndarray


class Netzwerk_Regressiv(Netzwerk):
    def gradient_descent(
        self,
        eingaben: ndarray,
        ziele: ndarray,
    ):
        geschichte = []
        kosten = 1.0
        while kosten > 0.005:
            # Vorwärtsdurchlauf: Berechnung der Vorhersagen
            vorhersagen = self.vorwaerts_durchlauf(eingaben)
            geschichte.append(vorhersagen)

            kosten = self.verlustfunktion.verlust(vorhersagen, ziele)

            # Rückwärtsdurchlauf: Berechnung der Gradienten
            self.rueckwaerts_durchlauf(vorhersagen, ziele)

            # Aktualisiere Gewichte und Bias basierend auf den Gradienten
            for schicht in self.schichten:
                # Update der Gewichte
                schicht.gewichte -= self.lern_rate * schicht.gewicht_gradient
                # Update der Bias-Wert
                schicht.bias -= self.lern_rate * schicht.bias_gradient

        return geschichte


def trainiere_netzwerk():
    eingaben = np.arange(0, 5, 0.1)
    eingaben = eingaben.reshape(len(eingaben), 1)
    ziele = np.sin(eingaben)

    netzwerk = Netzwerk_Regressiv(MittlererQuadratischerFehler, 0.1)
    netzwerk.schicht_hinzufügen(
        Schicht(1, 40),  # Eingabeschicht → versteckte Schicht
        ReLU(),  # Aktivierungsfunktion für die versteckte Schicht
    )
    netzwerk.schicht_hinzufügen(
        Schicht(40, 1),  # versteckte Schicht → Ausgabeschicht
        Linear(),  # Aktivierungsfunktion für die Ausgabeschicht
    )
    geschichte = netzwerk.gradient_descent(eingaben, ziele)

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
