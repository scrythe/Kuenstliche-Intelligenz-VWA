import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = 0.1 * np.random.randn(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.eingaben = eingaben
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias
        return selbst.ausgaben

    def rückwärts(selbst, dwerte):
        selbst.dgewichte = np.dot(selbst.eingaben.T, dwerte)
        selbst.dbias = np.sum(dwerte, axis=0, keepdims=True)
        selbst.deingaben = np.dot(dwerte, selbst.gewichte.T)


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.eingaben = eingaben
        selbst.ausgaben = np.maximum(0, eingaben)
        return selbst.ausgaben

    def rückwärts(selbst, dwerte):
        selbst.deingaben = dwerte * (selbst.eingaben > 0)


class Mean_Squared_Error:
    def vorwärts(ausgaben, lösungen):
        losses = np.power(ausgaben - lösungen, 2)
        return np.mean(losses)

    def rückwärts(ausgaben, lösungen):
        dloss = 2 * (ausgaben - lösungen)
        return dloss / len(dloss)


class SGD:
    def __init__(selbst, learn_rate):
        selbst.lern_rate = learn_rate

    def aktualisiere_parameter(selbst, schicht):
        schicht.gewichte -= schicht.dgewichte * selbst.lern_rate
        schicht.bias -= schicht.dbias * selbst.lern_rate


class Netzwerk:
    def __init__(selbst):
        lern_rate = 0.05
        selbst.schicht1 = Schicht(1, 64)
        selbst.aktivierung1 = ReLU()
        selbst.schicht2 = Schicht(64, 64)
        selbst.aktivierung2 = ReLU()
        selbst.schicht3 = Schicht(64, 1)
        selbst.loss_funktion = Mean_Squared_Error
        selbst.optimierer = SGD(lern_rate)

    def vorwärts_durchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        rohe_ausgaben_3 = selbst.schicht3.vorwärts(aktivierte_ausgaben_2)
        loss = selbst.loss_funktion.vorwärts(rohe_ausgaben_3, lösungen)
        return rohe_ausgaben_3, loss

    def trainieren(selbst, eingaben, lösungen):
        geschichte = []
        # loss = 100
        for _ in range(20000):
            # while loss > 6:
            ausgaben, loss = selbst.vorwärts_durchlauf(eingaben)
            # print(loss)
            selbst.rückwärts_durchlauf(eingaben, lösungen)
            selbst.optimierer.aktualisiere_parameter(selbst.schicht3)
            selbst.optimierer.aktualisiere_parameter(selbst.schicht2)
            selbst.optimierer.aktualisiere_parameter(selbst.schicht1)
            geschichte.append(ausgaben)
        return geschichte

    def rückwärts_durchlauf(selbst, eingaben, lösungen):
        dloss = selbst.loss_funktion.rückwärts(selbst.schicht3.ausgaben, lösungen)
        selbst.schicht3.rückwärts(dloss)
        selbst.aktivierung2.rückwärts(selbst.schicht3.deingaben)
        selbst.schicht2.rückwärts(selbst.aktivierung2.deingaben)
        selbst.aktivierung1.rückwärts(selbst.schicht2.deingaben)
        selbst.schicht1.rückwärts(selbst.aktivierung1.deingaben)


def f(x):
    return np.sin(x)


def f(x):
    return 0.05 * np.power(x, 3) - 0.5 * x


eingaben = np.arange(0, 5, 0.1).reshape(50, 1)
eingaben = np.arange(-5, 5, 0.5)
eingaben = eingaben.reshape(len(eingaben), 1)
# eingaben = np.arange(2, 10, 5).reshape(2, 1)
lösungen = f(eingaben)


fig, ax = plt.subplots(figsize=(10, 5))
(linie,) = ax.plot([], [], label="Approximation", color="orange")
(punkte,) = ax.plot([], [], "x", label="Ausgabe Daten", color="green")

# plt.plot(eingaben, lösungen, color="blue", label="Wahre Funktion Function")
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


linie.set_data(eingaben, geschichte[-1])

# framge_range = range(0, len(geschichte), 50)
# ani = animation.FuncAnimation(
#     fig, update, frames=framge_range, init_func=init, interval=100, repeat=False
# )

# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist="Magomed Alimkhanov"), bitrate=1800
# )
# ani.save("hm.mp4", writer)

plt.show()
