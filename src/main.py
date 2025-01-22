import numpy as np
from netzwerk import Netzwerk
from neuronale_schicht import Schicht
from aktivierungsfunktionen import ReLU, Softmax
from verlustfunktionen import Kreuzentropie
import lade_daten


# Trainings- und Testfunktionen
def trainiere_netzwerk():
    bilder, labels = lade_daten.lade_trainings_daten()
    netzwerk = Netzwerk(Kreuzentropie, 0.01)
    netzwerk.schicht_hinzufügen(
        Schicht(784, 20),  # Eingabeschicht → versteckte Schicht
        ReLU(),  # Aktivierungsfunktion für die versteckte Schicht
    )
    netzwerk.schicht_hinzufügen(
        Schicht(20, 10),  # Versteckte Schicht → Ausgabeschicht
        Softmax(),  # Aktivierungsfunktion für die Ausgabeschicht
    )
    netzwerk.SGD(bilder, labels, epochen=200, batch_groesse=64)


if __name__ == "__main__":
    trainiere_netzwerk()
