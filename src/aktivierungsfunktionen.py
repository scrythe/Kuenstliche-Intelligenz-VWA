import numpy as np
from numpy import ndarray


class AktivierungsFunktion:
    """
    Template-Aktivierungsfunktion für versteckten Schichten und Ausgabeschicht.
    """

    def vorwaerts(self, eingaben: ndarray) -> ndarray:
        """
        Führt die Vorwärtspropagierung durch: Aktiviert nur positive Werte speichert die aktuellen Eingaben für die Rückwärtspropagation.

        Parameter:
            eingaben (ndarray): Matrix der Eingaben (rohe Ausgaben der derzeitigen Schicht).

        Rückgabewert:
            ndarray: Matrix der aktivierten Ausgaben dieser Schicht.
        """
        raise NotImplementedError

    def rueckwaerts(self, verlust_gradient: ndarray) -> ndarray:
        """
        Führt die Rückwärtspropagation durch: Berechnet den Gradienten für die Eingaben.

        Parameter:
            verlust_gradient (ndarray): Der Gradient des Verlusts (Veränderung der aktivierten Ausgaben der aktuellen Schicht auf den Verlust) (dL/da).

        Rückgabewert:
            ndarray: Der berechnete Gradient für die Eingaben (Veränderung der rohen Ausgaben der aktuellen Schicht auf den Verlust) (dL/dz).
        """
        raise NotImplementedError


class ReLU(AktivierungsFunktion):
    """
    ReLU-Aktivierungsfunktion (Rectified Linear Unit) für die versteckten Schichten.

    Attribute:
        gespeicherte_eingaben (dnarray): Die gespeicherte Eingabematrix (Batch von Eingabevektoren).
    """

    def vorwaerts(self, eingaben):

        self.gespeicherte_eingaben = eingaben
        ausgaben = np.maximum(0, eingaben)
        return ausgaben

    def rueckwaerts(self, verlust_gradient):
        # Gradient der Eingaben berechnen (dL/dz)
        eingabe_gradient = verlust_gradient * (self.gespeicherte_eingaben > 0)
        return eingabe_gradient


class Softmax(AktivierungsFunktion):
    """
    Softmax-Aktivierungsfunktion für die Ausgabeschicht.
    """

    def vorwaerts(self, eingaben):
        # Normalisiert die Ausgaben in Wahrscheinlichkeiten
        exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
        summe = np.sum(exponierte_werte, axis=1, keepdims=True)
        ausgaben = exponierte_werte / summe
        return ausgaben

    def rueckwaerts(self, verlust_gradient):
        # Gibt die Gradienten direkt weiter (Softmax wird in Kombination mit Kreuzentropie verwendet)
        return verlust_gradient
