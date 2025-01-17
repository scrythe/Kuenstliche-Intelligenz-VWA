import numpy as np
import matplotlib.pyplot as plt


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))
        selbst.gradiantW = np.zeros((anzahl_eingaben, anzahl_neuronen))
        selbst.gradiantB = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias


class Sigmoid:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = 1 / (1 + np.exp(-eingaben))


def Mean_Squared_Error(ausgaben, lösungen):
    losses = np.power(ausgaben - lösungen, 2)
    avg_loss = np.sum(losses)
    return avg_loss


class Netzwerk:
    def __init__(selbst):
        anzahl_eingabe_neuronen = 1
        anzahl_versteckte_neuronen = 6
        anzahl_ausgabe_neuronen = 1

        selbst.versteckte_schicht = Schicht(
            anzahl_eingabe_neuronen, anzahl_versteckte_neuronen
        )
        selbst.versteckte_aktivierung = Sigmoid()
        selbst.ausgabe_schicht = Schicht(
            anzahl_versteckte_neuronen, anzahl_ausgabe_neuronen
        )
        selbst.ausgabe_aktivierung = Sigmoid()
        selbst.loss_funktion = Mean_Squared_Error

    def vorwärtspropagierung(selbst, eingaben):
        selbst.versteckte_schicht.vorwärts(eingaben)
        selbst.versteckte_aktivierung.vorwärts(selbst.versteckte_schicht.ausgaben)

        selbst.ausgabe_schicht.vorwärts(selbst.versteckte_aktivierung.ausgaben)
        selbst.ausgabe_aktivierung.vorwärts(selbst.ausgabe_schicht.ausgaben)
        return selbst.ausgabe_aktivierung.ausgaben

    def trainieren(selbst, line, eingaben, lösungen):
        while True:
            selbst.berechne_gradiant(eingaben, lösungen)
            selbst.anwende_gradiant()
            ausgaben = selbst.vorwärtspropagierung(eingaben)
            line.set_ydata(ausgaben)
            plt.pause(1/600)

    def anwende_gradiant(selbst):
        selbst.ausgabe_schicht.gewichte -= selbst.ausgabe_schicht.gradiantW * 0.1
        selbst.ausgabe_schicht.bias-= selbst.ausgabe_schicht.gradiantB * 0.1
        selbst.versteckte_schicht.gewichte -= selbst.versteckte_schicht.gradiantW * 0.1
        selbst.versteckte_schicht.bias-= selbst.versteckte_schicht.gradiantB * 0.1

    def berechne_gradiant(selbst, eingaben, lösungen):
        ausgaben = selbst.vorwärtspropagierung(eingaben)
        loss = selbst.loss_funktion(ausgaben, lösungen)
        selbst.berechne_gradiant_w_schicht(selbst.ausgabe_schicht,loss,eingaben,lösungen)
        selbst.berechne_gradiant_b_schicht(selbst.ausgabe_schicht,loss,eingaben,lösungen)
        selbst.berechne_gradiant_w_schicht(selbst.versteckte_schicht,loss,eingaben,lösungen)
        selbst.berechne_gradiant_b_schicht(selbst.versteckte_schicht,loss,eingaben,lösungen)

    def berechne_gradiant_w_schicht(selbst, schicht, loss, eingaben, lösungen):
        h = 0.001
        for i, _ in enumerate(schicht.gewichte):
            for j, _ in enumerate(schicht.gewichte[i]):
                schicht.gewichte[i][j] += h
                ausgaben = selbst.vorwärtspropagierung(eingaben)
                schicht.gewichte[i][j] -= h
                delta_loss = selbst.loss_funktion(ausgaben, lösungen) - loss
                schicht.gradiantW[i][j] = delta_loss / 2

    def berechne_gradiant_b_schicht(selbst, schicht, loss, eingaben, lösungen):
        h = 0.001
        for i, _ in enumerate(schicht.bias):
            schicht.bias[i] += h
            ausgaben = selbst.vorwärtspropagierung(eingaben)
            schicht.bias[i] -= h
            delta_loss = selbst.loss_funktion(ausgaben, lösungen) - loss
            schicht.gradiantB[i] = delta_loss / 2


x = np.arange(1, 5, 0.01)
x = (x).reshape(400, 1)
y = np.sin(x)
netzwerk = Netzwerk()
plt.ion()
plt.plot(x, y, color="blue", label="True Function")
plt.scatter(x, y, color="red", s=10, label="Data Points")
(line,) = plt.plot(x, np.zeros_like(x), label="Model Prediction", color="blue")
netzwerk.trainieren(line, x, y)
