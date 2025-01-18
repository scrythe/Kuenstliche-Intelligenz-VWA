import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias
        return selbst.ausgaben


class Sigmoid:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = 1 / (1 + np.exp(-eingaben))
        return selbst.ausgaben


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.maximum(0, eingaben)
        return selbst.ausgaben

    def ableiten(selbst, eingaben):
        return 1 * (eingaben > 0)


class Mean_Squared_Error:
    def berechnen(ausgaben, lösungen):
        losses = np.power(ausgaben - lösungen, 2)
        avg_loss = np.mean(losses)
        return avg_loss

    def ableiten(ausgaben, lösungen):
        return 2 * (ausgaben - lösungen)


class Netzwerk:
    def __init__(selbst):
        netzwerk_größe = [1, 3, 1]
        selbst.schicht1 = Schicht(netzwerk_größe[0], netzwerk_größe[1])
        selbst.schicht2 = Schicht(netzwerk_größe[1], netzwerk_größe[2])
        selbst.aktivierung1 = ReLU()
        selbst.aktivierung2 = ReLU()

        selbst.loss_funktion = Mean_Squared_Error

    def vorwärtsdurchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        return aktivierte_ausgaben_2

    def trainieren(selbst, eingaben, lösungen):
        geschichte = []
        loss = 100
        for _ in range(1000):
            # while loss > 6:
            ausgaben = selbst.vorwärtsdurchlauf(eingaben)
            loss = selbst.loss_funktion.berechnen(ausgaben, lösungen)
            # print(loss)
            geschichte.append(ausgaben)
            selbst.rückwärts(eingaben, lösungen)
        return geschichte

    def rückwärts(selbst, eingaben, lösungen):
        lern_rate = 0.0001
        ausgaben = selbst.vorwärtsdurchlauf(eingaben)
        loss = selbst.loss_funktion.berechnen(ausgaben, lösungen)
        dloss = selbst.loss_funktion.ableiten(selbst.aktivierung2.ausgaben, lösungen)
        drelu = dloss * selbst.aktivierung2.ableiten(selbst.schicht2.ausgaben)
        dweight = drelu * selbst.aktivierung1.ausgaben
        dweight = np.mean(dweight, axis=0, keepdims=True).T
        dbias = np.mean(drelu)

        # selbst.schicht2.gewichte -= dweight * lern_rate
        # selbst.schicht2.bias -= dbias * lern_rate

        d_ausgabe_1 = np.dot(drelu, selbst.schicht2.gewichte.T)
        # dweight = d_ausgabe_1 * eingaben
        # dweight = np.mean(dweight, axis=0)
        # dbias = np.mean(d_ausgabe_1, axis=0)

        selbst.aktivierung1.ausgaben += 0.001
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(selbst.aktivierung1.ausgaben)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        delta_loss = (
            selbst.loss_funktion.berechnen(aktivierte_ausgaben_2, lösungen) - loss
        )
        print(delta_loss / 0.001)
        print(np.mean(np.sum(d_ausgabe_1, axis=1)))

        # selbst.schicht1.gewichte -= dweight * lern_rate
        # selbst.schicht1.bias -= dbias * lern_rate


def f(x):
    return 3 * x**4


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


framge_range = range(0, len(geschichte), 1)
ani = animation.FuncAnimation(
    fig, update, frames=framge_range, init_func=init, interval=50, repeat=False
)

# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist="Magomed Alimkhanov"), bitrate=1800
# )
# ani.save("hm.gif", writer)

plt.show()
