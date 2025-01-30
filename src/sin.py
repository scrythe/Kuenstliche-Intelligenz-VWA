from neuronales_netzwerk.netzwerk import Netzwerk
from neuronales_netzwerk.schicht import Schicht
from neuronales_netzwerk.aktivierungsfunktionen import ReLU, Linear
from neuronales_netzwerk.verlustfunktionen import MittlererQuadratischerFehler
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


def trainiere_netzwerk(eingaben, ziele):
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
    return netzwerk,geschichte

