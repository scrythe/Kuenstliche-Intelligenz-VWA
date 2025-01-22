from verlustfunktionen import VerlustFunktion
from numpy import ndarray
from neuronale_schicht import Schicht
from aktivierungsfunktionen import AktivierungsFunktion


class Netzwerk:
    """
    Eine Klasse, die ein einfaches neuronales Netzwerk bestehend mit verschiedenen Schichten darstellt.

    Attribute:
        lern_rate (float): Lernrate für das Netzwerk.
        schichten (type): Die Schichten des Netzwerks.
        verlustfunktion (VerlustFunktion): Die Verlustfunktion des Netzwerks.
    """

    def __init__(self, verlustfunktion: VerlustFunktion, lern_rate: float):
        """
        Initialisiert das Netzwerk mit Schichten, Aktivierungsfunktionen und einer Kostenfunktion.

        Attribute:
            lern_rate (float): Lernrate für das Netzwerk.
            schichten (type): Die Schichten des Netzwerks.
            verlustfunktion (VerlustFunktion): Die Verlustfunktion des Netzwerks.
        """
        self.verlustfunktion = verlustfunktion
        self.lern_rate = lern_rate
        self.schichten: list[Schicht] = []
        self.aktivierungen: list[AktivierungsFunktion] = []

    def vorwaerts_durchlauf(self, eingaben: ndarray) -> ndarray:
        """
        Führt einen Vorwärtsdurchlauf durch alle Schichten und Aktivierungsfunktionen des Netzwerkes durch: Berechnet berechnet die Ausgaben jeder der Schichten schritt für schrittweise.

        Parameter:
            eingaben (ndarray): Matrix der Eingaben.

        Rückgabewert:
            ndarray: Matrix der vorhergesagten Ausgaben
        """
        for schicht, aktivierung in zip(self.schichten, self.aktivierungen):
            rohe_ausgaben = schicht.vorwaerts(eingaben)
            aktivierte_ausgaben = aktivierung.vorwaerts(rohe_ausgaben)
            # Ausgaben der Schicht werden zu Eingaben für die nächste Schicht
            eingaben = aktivierte_ausgaben
        return aktivierte_ausgaben

    def rueckwaerts_durchlauf(self, vorhersagen: ndarray, ziele: ndarray) -> ndarray:
        """
        Führt einen Rückwärtsdurchlauf durch alle Schichten, Aktivierungsfunktion und durch die Kostenfunktion des Netzwerk durch.
        Berechnet somit den Gradient für die Gewichte und Bias-Werte.

        Parameter:
            vorhersagen (ndarray): Matrix der Vorhersagen des Neuronalen Netzwerkes.
            ziele (ndarray): Matrix der Ziele / Wahren Werten.
        """
        # Veränderung der vorhergesagten Ausgaben auf den Verlust (dL/da)
        gradient = self.verlustfunktion.rueckwaerts(vorhersagen, ziele)
        for schicht, aktivierung in zip(self.schichten, self.aktivierungen):

            # Veränderung der rohen Ausgaben der aktuellen Schicht auf den Verlust) (dL/dz).
            gradient = aktivierung.rueckwaerts(gradient)

            # Veränderung der Gewichte, es Bias-Werte und den aktivierten Ausgaben der vorherigen Schicht auf den Verlust) (dL/dW) (dL/db) (dL/da)
            gradient = schicht.rueckwaerts(gradient)

    def SGD(self, eingaben: ndarray, ziele: ndarray, epochen=10, batch_groesse=32):
        """
        Trainieren des Netzwerkes über mehrere Epochen mit dem Optimierer Stochastic Gradient Descent

        Parameter:
            eingaben (ndarray): Matrix der Eingaben.
            ziele (ndarray): Matrix der Ziele / Wahren Werten.
            epochen (int): Anzahl der Epochen.
            batch_groesse (int): Größe des Batches / Menge an Trainingsdaten pro Batch.
        """
        for epoche in epochen:
            for start in range(0, len(eingaben), batch_groesse):
                # Unterteile die Trainingsdaten in Batches
                batch_eingaben = eingaben[start : start + batch_groesse]
                batch_ziele = ziele[start : start + batch_groesse]

                # Vorwärtsdurchlauf: Berechnung der Vorhersagen
                vorhersagen = self.vorwaerts(batch_eingaben)

                # # Berechne den Verlust (Fehler des Models)
                # verlust = self.verlustfunktion.verlust(vorhersagen, batch_ziele)

                # Rückwärtsdurchlauf: Berechnung der Gradienten
                self.rueckwaerts_durchlauf(batch_eingaben, batch_ziele)

                # Aktualisiere Gewichte und Bias basierend auf den Gradienten
                for schicht in self.schichten:
                    # Update der Gewichte
                    schicht.gewichte -= self.lern_rate * schicht.gewicht_gradient
                    # Update der Bias-Wert
                    schicht.bias -= self.lern_rate * schicht.bias_gradient

            # # Ausgabe des aktuellen Verlustes nach jeder Epoche
            # print(f"Epoche {epoche + 1}, Verlust: {verlust:.4f}")
