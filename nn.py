import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import gzip
import lade_daten


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))
        selbst.gradiantW = np.zeros((anzahl_eingaben, anzahl_neuronen))
        selbst.gradiantB = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias


def sigmoid(eingaben):
    return 1 / (1 + np.exp(-eingaben))


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.maximum(0, eingaben)


class Softmax:
    def vorwärts(selbst, eingaben):
        exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
        normalisierte_basis = np.sum(exponierte_werte, axis=1, keepdims=True)
        selbst.ausgaben = exponierte_werte / normalisierte_basis


def Categorical_Cross_Entropy(ausgaben, lösungen):
    ausgaben_clipped = np.clip(ausgaben, 1e-100, 1 - 1e-100)
    ausgaben_indexe = range(len(ausgaben))
    gewählte_ausgaben = ausgaben_clipped[ausgaben_indexe, lösungen]
    losses = -np.log(gewählte_ausgaben)
    avg_loss = np.mean(losses)
    return avg_loss


class Netzwerk:
    def __init__(selbst):
        anzahl_eingabe_neuronen = 784
        anzahl_versteckte_neuronen = 20
        anzahl_ausgabe_neuronen = 10

        selbst.versteckte_schicht = Schicht(
            anzahl_eingabe_neuronen, anzahl_versteckte_neuronen
        )
        selbst.versteckte_aktivierung = ReLU()
        selbst.ausgabe_schicht = Schicht(
            anzahl_versteckte_neuronen, anzahl_ausgabe_neuronen
        )
        selbst.ausgabe_aktivierung = Softmax()
        selbst.loss_funktion = Categorical_Cross_Entropy

    def vorwärtspropagierung(selbst, eingaben):
        selbst.versteckte_schicht.vorwärts(eingaben)
        selbst.versteckte_aktivierung.vorwärts(selbst.versteckte_schicht.ausgaben)

        selbst.ausgabe_schicht.vorwärts(selbst.versteckte_aktivierung.ausgaben)
        selbst.ausgabe_aktivierung.vorwärts(selbst.ausgabe_schicht.ausgaben)

    def trainieren(selbst, eingaben, lösungen):
        for _ in range(20):
            selbst.berechne_gradiant(eingaben, lösungen)
            selbst.ausgabe_schicht.gewichte -= selbst.ausgabe_schicht.gradiantW * 1
            selbst.ausgabe_schicht.bias -= selbst.ausgabe_schicht.gradiantB * 1
            selbst.versteckte_schicht.gewichte -= (
                selbst.versteckte_schicht.gradiantW * 1
            )
            selbst.versteckte_schicht.bias -= selbst.versteckte_schicht.gradiantB * 1

    def berechne_gradiant(selbst, eingaben, lösungen):
        h = 1
        selbst.vorwärtspropagierung(eingaben)
        loss = selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
        print(loss)
        for i, neuron in enumerate(selbst.ausgabe_schicht.gewichte):
            for j, _ in enumerate(neuron):
                neuron[j] += h
                selbst.vorwärtspropagierung(eingaben)
                delta_loss = (
                    selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
                    - loss
                )
                neuron[j] -= h
                selbst.ausgabe_schicht.gradiantW[i][j] = delta_loss / h

        for i, _ in enumerate(selbst.ausgabe_schicht.bias):
            selbst.ausgabe_schicht.bias[i] += h
            selbst.vorwärtspropagierung(eingaben)
            delta_loss = (
                selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
                - loss
            )
            selbst.ausgabe_schicht.bias[i] -= h
            selbst.ausgabe_schicht.gradiantB[i] = delta_loss / h

        for i, neuron in enumerate(selbst.versteckte_schicht.gewichte):
            for j, _ in enumerate(neuron):
                neuron[j] += h
                selbst.vorwärtspropagierung(eingaben)
                delta_loss = (
                    selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
                    - loss
                )
                neuron[j] -= h
                selbst.versteckte_schicht.gradiantW[i][j] = delta_loss / h

        for i, _ in enumerate(selbst.versteckte_schicht.bias):
            selbst.ausgabe_schicht.bias[i] += h
            selbst.vorwärtspropagierung(eingaben)
            delta_loss = (
                selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
                - loss
            )
            selbst.ausgabe_schicht.bias[i] -= h
            selbst.versteckte_schicht.gradiantB[i] = delta_loss / h


def test():
    bilder, beschriftungen = lade_daten.lade_test_daten()
    netzwerk = Netzwerk()
    index = int(input("Zahl von 0 bis 59999: "))
    plt.imshow(bilder[index].reshape(28, 28), cmap="Greys")
    vorhergesagte_werte = netzwerk.vorwärtspropagierung(bilder[5])
    plt.title(vorhergesagte_werte[0])
    plt.show()


def trainieren():
    bilder, beschriftungen = lade_daten.lade_test_daten()
    netzwerk = Netzwerk()
    netzwerk.trainieren(bilder[0:20], beschriftungen[0:20])


trainieren()
