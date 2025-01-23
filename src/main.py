import numpy as np
from netzwerk import Netzwerk
from neuronale_schicht import Schicht
from aktivierungsfunktionen import ReLU, Softmax
from verlustfunktionen import Kreuzentropie
import lade_daten
import matplotlib.pyplot as plt


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
    lade_daten.speicher_netzwerk(netzwerk)


def teste_netzwerk():
    bilder, beschriftungen = lade_daten.lade_test_daten()
    netzwerk = lade_daten.lade_netzwerk()
    vorhersagen = netzwerk.vorwaerts_durchlauf(bilder)
    kost = netzwerk.verlustfunktion.verlust(vorhersagen, beschriftungen)
    genauigkeit = netzwerk.berechne_genauigkeit(vorhersagen, beschriftungen)
    print(
        "Genauigkeit: ",
        genauigkeit,
        "Kost: ",
        kost,
    )


def interaktives_testen():
    bilder, beschriftungen = lade_daten.lade_test_daten()
    netzwerk = lade_daten.lade_netzwerk()
    while True:
        index = int(input("Zahl von 0 bis 9999: "))
        vorhersagen = netzwerk.vorwaerts_durchlauf(bilder[index])
        vorhersage = np.argmax(vorhersagen)
        ziel = np.argmax(beschriftungen[index])
        plt.imshow(bilder[index].reshape(28, 28), cmap="Greys")
        plt.title(f"Vorhersage: {vorhersage}, Eigentliche Zahl: {ziel}")
        plt.show()


if __name__ == "__main__":
    # trainiere_netzwerk()
    # teste_netzwerk()
    interaktives_testen()
