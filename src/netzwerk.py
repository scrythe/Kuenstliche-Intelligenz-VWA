from verlustfunktionen import VerlustFunktion
from numpy import ndarray
from neuronale_schicht import Schicht
from aktivierungsfunktionen import AktivierungsFunktion
import numpy as np


class Netzwerk:
    """
    Eine Klasse, die ein einfaches neuronales Netzwerk bestehend mit verschiedenen Schichten darstellt.

    Attribute:
        lern_rate (float): Lernrate für das Netzwerk.
        schichten (type): Die Schichten des Netzwerks.
        verlustfunktion (VerlustFunktion): Die Verlustfunktion des Netzwerks.
    """

    def __init__(
        self,
        verlustfunktion: VerlustFunktion,
        lern_rate: float,
    ):
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

    def schicht_hinzufügen(self, schicht: Schicht, aktivierung: AktivierungsFunktion):
        """
        Fügt Schicht mit der zuständigen Aktivierungsfunktion hinzu
        """
        self.schichten.append(schicht)
        self.aktivierungen.append(aktivierung)

    def vorwaerts_durchlauf(self, eingaben: ndarray) -> ndarray:
        """
        Führt einen Vorwärtsdurchlauf durch alle Schichten und Aktivierungsfunktionen des Netzwerkes durch: Berechnet berechnet die Ausgaben jeder der Schichten schritt für schrittweise.

        Parameter:
            eingaben (ndarray): Matrix der Eingaben.

        Rückgabewert:
            ndarray: Matrix der vorhergesagten Ausgaben
        """
        aktivierte_ausgaben = []
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
        # Rückwärts berechnet, von Ausgabeschicht zu Eingabeschicht
        for schicht, aktivierung in zip(
            reversed(self.schichten), reversed(self.aktivierungen)
        ):
            # Veränderung der rohen Ausgaben der aktuellen Schicht auf den Verlust) (dL/dz).
            gradient = aktivierung.rueckwaerts(gradient)

            # Veränderung der Gewichte, es Bias-Werte und den aktivierten Ausgaben der vorherigen Schicht auf den Verlust) (dL/dW) (dL/db) (dL/da)
            gradient = schicht.rueckwaerts(gradient)

    def berechne_genauigkeit(selbst, vorhersagen, ziele) -> float:
        """
        Berechnet die Genauigkeit der Vorhersagen im Vergleich zu den tatsächlichen Zielen.

        Parameter:
            vorhersagen (ndarray): Matrix der Vorhersagen, wobei jede Zeile eine Vorhersage darstellt.
            ziele (ndarray): Matrix der tatsächlichen Zielen

        Rückgabewert:
            float: Die Genauigkeit der Vorhersagen als Mittelwert der korrekten Vorhersagen.
        """
        ziele = np.argmax(ziele, axis=1)
        vorhersagen = np.argmax(vorhersagen, axis=1)
        vergleich = vorhersagen == ziele
        genauigkeit = np.mean(vergleich)
        return genauigkeit

    def daten_vermischen(selbst, eingaben, ziele) -> tuple[ndarray, ndarray]:
        """
        Vermischt die Eingabedaten und die Zielwerte

        Parameter:
            eingaben (ndarray): Matrix von Eingabedaten.
            ziele (ndarray): Matrix von Zielwerten.

        Rückgabewert:
            tuple: Ein Tupel, das die vermischten Eingabedaten und Zielwerte enthält.
        """
        schlüssel = np.array(range(eingaben.shape[0]))
        np.random.shuffle(schlüssel)
        return (eingaben[schlüssel], ziele[schlüssel])

    def SGD(self, eingaben: ndarray, ziele: ndarray, epochen, batch_groesse):
        """
        Trainieren des Netzwerkes über mehrere Epochen mit dem Optimierer Stochastic Gradient Descent

        Parameter:
            eingaben (ndarray): Matrix der Eingaben.
            ziele (ndarray): Matrix der Ziele / Wahren Werten.
            epochen (int): Anzahl der Epochen.
            batch_groesse (int): Größe des Batches / Menge an Trainingsdaten pro Batch.
        """
        genauigkeiten = []
        verluste = []
        anzahl_trainings_daten = len(ziele)
        for epoche in range(epochen):
            eingaben, ziele = self.daten_vermischen(eingaben, ziele)
            for start in range(0, len(eingaben), batch_groesse):
                # Unterteile die Trainingsdaten in Batches
                batch_eingaben = eingaben[start : start + batch_groesse]
                batch_ziele = ziele[start : start + batch_groesse]

                # Vorwärtsdurchlauf: Berechnung der Vorhersagen
                vorhersagen = self.vorwaerts_durchlauf(batch_eingaben)

                # Berechne den Verlust (Fehler des Models)
                verlust = self.verlustfunktion.verlust(vorhersagen, batch_ziele)
                verluste.append(verlust)

                # Berechne die Genauigkeit des Modells (Wie viele korrekte Antworten)
                genauigkeit = self.berechne_genauigkeit(vorhersagen, batch_ziele)
                genauigkeiten.append(genauigkeit)

                # Rückwärtsdurchlauf: Berechnung der Gradienten
                self.rueckwaerts_durchlauf(vorhersagen, batch_ziele)

                # Aktualisiere Gewichte und Bias basierend auf den Gradienten
                for schicht in self.schichten:
                    # Update der Gewichte
                    schicht.gewichte -= self.lern_rate * schicht.gewicht_gradient
                    # Update der Bias-Wert
                    schicht.bias -= self.lern_rate * schicht.bias_gradient

            # Ausgabe des aktuellen Verlustes nach jeder Epoche
            print(
                f"Epoche {epoche + 1}, Verlust: {np.mean(verluste):.4f}, Genauigkeit: {np.mean(genauigkeiten):.4f}"
            )
