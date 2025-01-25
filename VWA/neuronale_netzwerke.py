eingaben = [0.3, 0.6]
gewichte = [0.8, 0.2]
bias = 4

ausgabe = eingaben[0] * gewichte[0] + eingaben[1] * gewichte[1] + bias

#####

eingaben = [1.2, 3.2]
gewichte1 = [0.8, 1.3]  # Gewichte zwischen des ersten Ausgabeneuron
gewichte2 = [3.1, 1.6]  # Gewichte zwischen des zweiten Ausgabeneuron

bias1 = 4
bias2 = 3

ausgabe1 = eingaben[0] * gewichte1[0] + eingaben[1] * gewichte1[1] + bias1
ausgabe2 = eingaben[0] * gewichte2[0] + eingaben[1] * gewichte2[1] + bias2

#####

import numpy as np

eingaben = [1.2, 3.2]
gewichte = [
    [0.8, 1.3],
    [3.1, 1.6],
]
biases = [4, 3]

ausgabe = np.dot(gewichte, eingaben) + biases

# print(ausgabe)

#####

eingaben = [
    [1.2, 3.2, 4.2],
    [3.2, 1.2, 2.2],
    [-4.2, 0.2, 0.5],
    [3.1, 2.2, 0.5],
]

gewichte = [
    [0.8, 1.3, 3.2],
    [3.1, 1.6, 0.4],
]

transponierte_gewichte = np.array(gewichte).T
ausgabe = np.dot(eingaben, transponierte_gewichte) + biases

# print(ausgabe)

#####


class Schicht:
    def __init__(self, anzahl_eingaben, anzahl_neuronen):
        self.gewichte = 0.1 * np.random.randn(anzahl_eingaben, anzahl_neuronen)
        self.bias = 0.1 * np.random.randn(1, anzahl_neuronen)

    def vorwaerts(self, eingaben):
        self.gespeicherte_eingaben = eingaben
        ausgaben = np.dot(eingaben, self.gewichte) + self.bias
        return ausgaben


schicht1 = Schicht(3, 4)
schicht2 = Schicht(4, 5)

ausgaben1 = schicht1.vorwaerts(eingaben)
ausgaben2 = schicht2.vorwaerts(ausgaben1)
print(ausgaben2)

#####


#
ausgabe = 5
#

max(0, ausgabe)


#####


class Sigmoid:
    def vorwaerts(self, eingaben):
        self.gespeicherte_ausgaben = 1 / (1 + np.exp(-eingaben))
        return self.gespeicherte_ausgaben


schicht1 = Schicht(3, 4)
aktivierung1 = Sigmoid()

rohe_ausgaben = schicht1.vorwaerts(eingaben)
aktivierte_ausgaben = aktivierung1.vorwaerts(rohe_ausgaben)
print(aktivierte_ausgaben)

#####


class ReLU:
    def vorwaerts(self, eingaben):
        self.gespeicherte_eingaben = eingaben
        ausgaben = np.maximum(0, eingaben)
        return ausgaben


#####


import math

ausgaben = [0.8, 1.3, 3.1, 1.6]
exponierte_werte = []

for ausgabe in ausgaben:
    exponierte_werte.append(math.e**ausgabe)

print(exponierte_werte)


######

normalisierte_basis = sum(exponierte_werte)
normalisierte_werte = []

for ausgabe in exponierte_werte:
    normalisierte_werte.append(ausgabe / normalisierte_basis)

print(normalisierte_werte)


######
class Softmax:
    def vorwaerts(self, eingaben):
        # Normalisiert die Ausgaben in Wahrscheinlichkeiten
        exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
        summe = np.sum(exponierte_werte, axis=1, keepdims=True)
        ausgaben = exponierte_werte / summe
        return ausgaben


#####
from lade_daten import lade_test_daten
import matplotlib.pyplot as plt


class Netzwerk:
    def __init__(
        self,
    ):
        self.schichten = []
        self.aktivierungen = []

    def schicht_hinzufügen(self, schicht, aktivierung):
        self.schichten.append(schicht)
        self.aktivierungen.append(aktivierung)

    def vorwaerts_durchlauf(self, eingaben):
        for schicht, aktivierung in zip(self.schichten, self.aktivierungen):
            rohe_ausgaben = schicht.vorwaerts(eingaben)
            aktivierte_ausgaben = aktivierung.vorwaerts(rohe_ausgaben)
            # Ausgaben der Schicht werden zu Eingaben für die nächste Schicht
            eingaben = aktivierte_ausgaben
        return aktivierte_ausgaben


netzwerk = Netzwerk()
netzwerk.schicht_hinzufügen(
    Schicht(784, 20),  # Eingabeschicht → versteckte Schicht
    ReLU(),  # Aktivierungsfunktion für die versteckte Schicht
)
netzwerk.schicht_hinzufügen(
    Schicht(20, 10),  # Versteckte Schicht → Ausgabeschicht
    Softmax(),  # Aktivierungsfunktion für die Ausgabeschicht
)
bilder, beschriftungen = lade_test_daten()
index = int(input("Zahl von 0 bis 59999: "))
bild = bilder[index]
plt.imshow(bild.reshape(28, 28), cmap="Greys")
ergebnisse = netzwerk.vorwaerts_durchlauf(bilder[index])
plt.title(ergebnisse[0])
plt.show()
