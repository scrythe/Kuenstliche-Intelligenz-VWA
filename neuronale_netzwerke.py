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
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias


schicht1 = Schicht(3, 4)
schicht2 = Schicht(4, 5)

schicht1.vorwärts(eingaben)
schicht2.vorwärts(schicht1.ausgaben)
print(schicht2.ausgaben)

#####


#
ausgabe = 5
#

max(0, ausgabe)


#####


class Sigmoid:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = 1 / (1 + np.exp(-eingaben))


schicht1 = Schicht(3, 4)
aktivierung1 = Sigmoid()

schicht1.vorwärts(eingaben)
aktivierung1.vorwärts(schicht1.ausgaben)
print(aktivierung1.ausgaben)

#####


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.maximum(0, eingaben)


#####

#
ausgaben = [0.8, 1.3, 3.1, 1.6]
#

import math

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


class SoftMax:
    def vorwärts(selbst, eingaben):
        exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
        normalisierte_basis = np.sum(exponierte_werte, axis=1, keepdims=True)
        selbst.ausgaben = exponierte_werte / normalisierte_basis
