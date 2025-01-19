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
        netzwerk_größe = [1, 8, 9, 1]
        selbst.schicht1 = Schicht(netzwerk_größe[0], netzwerk_größe[1])
        selbst.schicht2 = Schicht(netzwerk_größe[1], netzwerk_größe[2])
        selbst.schicht3 = Schicht(netzwerk_größe[2], netzwerk_größe[3])
        selbst.aktivierung1 = ReLU()
        selbst.aktivierung2 = ReLU()

        selbst.loss_funktion = Mean_Squared_Error

    def vorwärtsdurchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        rohe_ausgaben_3 = selbst.schicht3.vorwärts(aktivierte_ausgaben_2)
        return rohe_ausgaben_3

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
        lern_rate = 0.1
        ausgaben = selbst.vorwärtsdurchlauf(eingaben)
        loss = selbst.loss_funktion.berechnen(ausgaben, lösungen)

        dloss = selbst.loss_funktion.ableiten(selbst.schicht3.ausgaben, lösungen)
        dweight1 = dloss * selbst.aktivierung2.ausgaben
        dweight1 = np.mean(dweight1, axis=0, keepdims=True).T
        dbias1 = np.mean(dloss)

        d_ausgabe_1 = np.dot(dloss, selbst.schicht3.gewichte.T)
        d_relu_2 = d_ausgabe_1 * selbst.aktivierung2.ableiten(selbst.schicht2.ausgaben)
        d_relu_2 = np.mean(d_relu_2, axis=0, keepdims=True)
        dweight2 = np.dot(
            d_relu_2.T, np.mean(selbst.aktivierung1.ausgaben, axis=0, keepdims=True)
        )
        dweight2 = np.sum(dweight2, axis=0)
        dbias2 = np.sum(d_relu_2, axis=1)

        selbst.schicht2.bias += 0.0001
        ausgaben = selbst.vorwärtsdurchlauf(eingaben)
        delta_loss = selbst.loss_funktion.berechnen(ausgaben, lösungen) - loss
        print(delta_loss / 0.0001)
        print(np.sum(dbias2))

        d_ausgabe_2 = np.dot(d_ausgabe_1, selbst.schicht2.gewichte.T)
        d_relu_3 = d_ausgabe_2 * selbst.aktivierung1.ableiten(selbst.schicht2.ausgaben)
        dweight3 = d_relu_3 * eingaben
        dweight3 = np.mean(dweight3, axis=0)
        dbias3 = np.mean(d_relu_3, axis=0)

        selbst.schicht3.gewichte -= dweight1 * lern_rate
        selbst.schicht3.bias -= dbias1 * lern_rate
        selbst.schicht2.gewichte -= dweight2 * lern_rate
        selbst.schicht2.bias -= dbias2 * lern_rate
        selbst.schicht1.gewichte -= dweight3 * lern_rate
        selbst.schicht1.bias -= dbias3 * lern_rate


def f(x):
    return 3 * x**2


eingaben = np.arange(0, 5, 0.5).reshape(10, 1)
eingaben = np.arange(2, 10, 5).reshape(2, 1)
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
