import numpy as np
from numpy import ndarray


class Schicht:
    """
    Eine Klasse, die eine Schicht eines neuronalen Netzwerkes (Eingabeschict,
    verstecke Schicht oder Ausgabeschicht) repräsentiert.

    Attribute:
        gewichte (dnarray): Die Gewichtsmatrix der Schicht.
        bias (dnarray): Der Biasvektor der Schicht.
        gespeicherte_eingaben (dnarray): Die gespeicherte Eingabematrix (Batch von Eingabevektoren).

    """

    def __init__(self, anzahl_eingaben: int, anzahl_neuronen: int):
        """ "
        Initialisiert die Schicht mit zufälligen Gewichten und den Bias-Werten.

        Parameter:
            anzahl_eingaben (int): Anzahl der Eingabeneuronen.
            anzahl_neuronen (int): Anzahl der Neuronen in dieser Schicht.
        """
        self.gewichte = np.random.randn(
            anzahl_eingaben, anzahl_neuronen
        )  # Zufällige Gewichte
        self.bias = np.random.randn(1, anzahl_neuronen)  # Zufällige Bias-Werte

    def vorwaerts(self, eingaben: ndarray) -> ndarray:
        """
        Führt die Vorwärtspropagierung durch: Berechnet die rohen Ausgaben der Schicht basierend auf die Eingaben
        und speichert die aktuellen Eingaben für die Rückwärtspropagation.

        Parameter:
            eingaben (ndarray): Matrix der Eingaben (z. B. Ausgabe der vorherigen Schicht).

        Rückgabewert:
            ndarray: Die berechnete rohe Ausgabematrix dieser Schicht.
        """
        self.gespeicherte_eingaben = eingaben
        ausgaben = np.dot(eingaben, self.gewichtsmatrix) + self.bias_vektor
        return ausgaben

    def rueckwaerts(self, verlustgradient: ndarray) -> ndarray:
        """
        Führt die Rückwärtspropagation durch: Berechnet die Gradienten für Gewichte, Bias und Eingaben.

        Parameter:
            verlustgradient (ndarray): Der Gradient des Verlusts (Veränderung der rohen Ausgaben der aktuellen Schicht auf den Verlust) (dL/dz).

        Rückgabewert:
            ndarray: Der berechnete Gradient für die Eingaben (Veränderung der aktivierten Ausgaben der vorherigen Schicht auf den Verlust) (dL/da).
        """
        # Gradient der Gewichte berechnen (dL/dW)
        self.gewicht_gradient = np.dot(self.gespeicherte_eingaben.T, verlustgradient)

        # Gradient der Bias-Werte berechnen (dL/db)
        self.bias_gradient = np.sum(verlustgradient, axis=0, keepdims=True)

        # Gradient der Eingaben berechnen (dL/da)
        eingabe_gradient = np.dot(verlustgradient, self.gewicht_matrix.T)
        return eingabe_gradient
