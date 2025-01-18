import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

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
        geschichte = []
        loss = 100
        # for _ in range(500):
        while loss > 6:
            ausgaben = selbst.vorwärtspropagierung(eingaben)
            loss = selbst.loss_funktion(ausgaben, lösungen)
            geschichte.append(ausgaben)
            selbst.rückwärts(eingaben, lösungen)
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


def f(x):
    return 3 * x * x


eingaben = np.arange(0, 5, 0.5).reshape(10, 1)
lösungen = f(eingaben)


fig, ax = plt.subplots(figsize=(10, 5))
(linie,) = ax.plot([], [], label="Approximation", color="orange")
(punkte,) = ax.plot([], [], "x", label="Ausgabe Daten", color="green")

plt.plot(eingaben, lösungen, color="blue", label="Wahre Funktion Function")
plt.plot(eingaben, lösungen, "x", color="red", label="Trainings Daten")


netzwerk = Netzwerk()
geschichte = netzwerk.trainieren(eingaben, lösungen)


def init():
    linie.set_xdata(eingaben)
    punkte.set_xdata(eingaben)


def update(epoche):
    linie.set_ydata(geschichte[epoche])
    punkte.set_ydata(geschichte[epoche])
    ax.set_title(f"Epoche {epoche}")


framge_range = range(0, len(geschichte), 5)

ani = animation.FuncAnimation(
    fig, update, frames=framge_range, init_func=init, interval=50, repeat=False
)

# writer = animation.PillowWriter(
#     fps=15, metadata=dict(artist="Magomed Alimkhanov"), bitrate=1800
# )
# ani.save("hm.gif", writer)

plt.show()
