import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = 0.1 * np.random.randn(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.eingaben = eingaben
        return np.dot(eingaben, selbst.gewichte) + selbst.bias

    def rückwärts(selbst, dkost_dausgaben):
        selbst.dgewichte = np.dot(selbst.eingaben.T, dkost_dausgaben)
        selbst.dbias = np.sum(dkost_dausgaben, axis=0, keepdims=True)
        selbst.deingaben = np.dot(dkost_dausgaben, selbst.gewichte.T)


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.eingaben = eingaben
        return np.maximum(0, eingaben)

    def ableiten_rohe_ausgaben(selbst, dkost_daktivierte_ausgaben):
        return dkost_daktivierte_ausgaben * (selbst.eingaben > 0)


class Mean_Squared_Error:
    def berechnen(ausgaben, lösungen):
        losses = np.power(ausgaben - lösungen, 2)
        kost = np.mean(losses)
        return kost

    def ableiten_auf_prädiktionen(prädiktionen, lösungen):
        dlosses_dausgaben = 2 * (prädiktionen - lösungen)
        dkost_dausgaben = dlosses_dausgaben / len(prädiktionen)
        return dkost_dausgaben


class Netzwerk:
    def __init__(selbst):
        selbst.lern_rate = 0.05
        selbst.schicht1 = Schicht(1, 64)
        selbst.aktivierung1 = ReLU()
        selbst.schicht2 = Schicht(64, 64)
        selbst.aktivierung2 = ReLU()
        selbst.schicht3 = Schicht(64, 1)
        selbst.kost_funktion = Mean_Squared_Error

    def vorwärts_durchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        rohe_ausgaben_3 = selbst.schicht3.vorwärts(aktivierte_ausgaben_2)
        return rohe_ausgaben_3

    def rückwärts_durchlauf(selbst, prädiktionen, lösungen):
        dkost_dprädiktionen = selbst.kost_funktion.ableiten_auf_prädiktionen(
            prädiktionen, lösungen
        )

        selbst.schicht3.rückwärts(dkost_dprädiktionen)
        dkost_daktivierte_ausgaben_2 = selbst.schicht3.deingaben
        dkost_drohe_ausgaben_2 = selbst.aktivierung2.ableiten_rohe_ausgaben(
            dkost_daktivierte_ausgaben_2
        )

        selbst.schicht2.rückwärts(dkost_drohe_ausgaben_2)
        dkost_daktivierte_ausgaben_2 = selbst.schicht2.deingaben
        dkost_drohe_ausgaben_2 = selbst.aktivierung2.ableiten_rohe_ausgaben(
            dkost_daktivierte_ausgaben_2
        )

        selbst.schicht1.rückwärts(dkost_drohe_ausgaben_2)

    def SGD_optimierer(selbst):
        for schicht in [selbst.schicht1, selbst.schicht2, selbst.schicht3]:
            schicht.gewichte -= schicht.dgewichte * selbst.lern_rate
            schicht.bias -= schicht.dbias * selbst.lern_rate

    def trainieren(selbst, eingaben, lösungen):
        geschichte = []
        kost = 1
        # for _ in range(20000):
        while kost > 0.01:
            prädiktionen = selbst.vorwärts_durchlauf(eingaben)
            selbst.rückwärts_durchlauf(prädiktionen, lösungen)
            selbst.SGD_optimierer()
            kost = selbst.kost_funktion.berechnen(prädiktionen, lösungen)
            # print(kost)
            geschichte.append(prädiktionen)
        return geschichte


def f(x):
    return 0.05 * np.power(x, 3) - 0.5 * x


eingaben = np.arange(-5, 5, 0.1)
eingaben = eingaben.reshape(len(eingaben), 1)
lösungen = f(eingaben)


fig, ax = plt.subplots()
(linie,) = ax.plot([], [], label="Approximierte Funktion", color="orange")
(punkte,) = ax.plot([], [], "x", label="Aproximierte Daten", color="green")

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


plt.legend()

framge_range = range(0, len(geschichte), 10)
ani = animation.FuncAnimation(
    fig, update, frames=framge_range, init_func=init, interval=20, repeat=False
)

writer = animation.FFMpegWriter(
    fps=15, metadata=dict(artist="Magomed Alimkhanov"), bitrate=1800
)
ani.save("hm.mp4", writer)

plt.show()
