import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = [[2]]
        selbst.bias = [[-2]]

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias


class Sigmoid:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = 1 / (1 + np.exp(-eingaben))


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.maximum(0, eingaben)

    def ableitung(selbst, eingaben):
        return 1 * (eingaben > 0)


def Mean_Squared_Error(ausgaben, lösungen):
    losses = np.power(ausgaben - lösungen, 2)
    avg_loss = np.mean(losses)
    return avg_loss


def Mean_Squared_Error_Ableitung(ausgaben, lösungen):
    return 2 * (ausgaben - lösungen)


class Netzwerk:
    def __init__(selbst):
        anzahl_eingabe_neuronen = 1
        anzahl_ausgabe_neuronen = 1

        selbst.ausgabe_schicht = Schicht(
            anzahl_eingabe_neuronen, anzahl_ausgabe_neuronen
        )
        selbst.ausgabe_aktivierung = ReLU()
        selbst.loss_funktion = Mean_Squared_Error
        selbst.loss_ableitung = Mean_Squared_Error_Ableitung

    def vorwärtspropagierung(selbst, eingaben):
        selbst.ausgabe_schicht.vorwärts(eingaben)
        selbst.ausgabe_aktivierung.vorwärts(selbst.ausgabe_schicht.ausgaben)
        return selbst.ausgabe_aktivierung.ausgaben

    def trainieren(selbst, eingaben, lösungen):
        loss = 1
        geschichte = []
        # losses = []
        while loss > 0.01:
            ausgaben = selbst.vorwärtspropagierung(eingaben)
            selbst.rückwärts(eingaben, lösungen)
            geschichte.append(ausgaben)
            loss = selbst.loss_funktion(selbst.ausgabe_aktivierung.ausgaben, lösungen)
            print(loss)
        return geschichte

    def rückwärts(selbst, eingaben, lösungen):
        dloss = selbst.loss_ableitung(selbst.ausgabe_aktivierung.ausgaben, lösungen)
        drelu = dloss * selbst.ausgabe_aktivierung.ableitung(
            selbst.ausgabe_aktivierung.ausgaben
        )
        dweight = drelu * eingaben
        dweight = np.mean(dweight)
        dbias = np.mean(drelu)

        selbst.ausgabe_schicht.gewichte -= dweight * 0.1
        selbst.ausgabe_schicht.bias -= dbias * 0.1
        # plt.scatter(
        #     eingaben,
        #     dweight,
        #     color="green",
        # )


def f(x):
    return 3 * x + 2


x = np.arange(0, 5, 0.5)
x = (x).reshape(10, 1)
y = f(x)

fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.plot([], [], label="NN Approximation", color="orange")

plt.plot(x, y, color="blue", label="True Function")
plt.scatter(x, y, color="red", s=10, label="Data Points")

netzwerk = Netzwerk()
geschichte = netzwerk.trainieren(x, y)


def update(epoche):
    line.set_data(x, geschichte[epoche])
    ax.set_title(f"Epoche {epoche}")
    return (line,)


ani = animation.FuncAnimation(fig, update, frames=(len(geschichte)), interval=50)

plt.show()
