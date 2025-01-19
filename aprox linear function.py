import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
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
        lern_rate = 0.1
        selbst.schicht1 = Schicht(1, 8)
        selbst.aktivierung1 = ReLU()
        selbst.schicht2 = Schicht(8, 8)
        selbst.aktivierung2 = ReLU()
        selbst.schicht3 = Schicht(8, 1)
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
        for _ in range(1000):
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
        loss = selbst.loss_funktion.vorwärts(selbst.schicht3.ausgaben, lösungen)
        selbst.schicht3.rückwärts(dloss)
        selbst.aktivierung2.rückwärts(selbst.schicht3.deingaben)
        selbst.schicht2.rückwärts(selbst.aktivierung2.deingaben)
        selbst.aktivierung1.rückwärts(selbst.schicht2.deingaben)
        selbst.schicht1.rückwärts(selbst.aktivierung1.deingaben)


def f(x):
    return 3 * x**2


eingaben = np.arange(0, 5, 0.5).reshape(10, 1)
# eingaben = np.arange(2, 10, 5).reshape(2, 1)
lösungen = f(eingaben)


fig, ax = plt.subplots(figsize=(10, 5))
(linie,) = ax.plot([], [], label="Approximation", color="orange")
(punkte,) = ax.plot([], [], "x", label="Ausgabe Daten", color="green")

plt.plot(eingaben, lösungen, color="blue", label="Wahre Funktion Function")
plt.plot(eingaben, lösungen, "x", color="red", label="Trainings Daten")


netzwerk = Netzwerk()


def optimal():
    netzwerk.schicht1.gewichte = [
        [1.2072, -0.7405, 1.3646, 0.5692, 1.6801, -0.1454, -0.4372, 1.1967]
    ]

    netzwerk.schicht1.bias = [
        [
            -2.0098,
            -0.0290,
            0.7159,
            0.5741,
            0.7426,
            -0.4148,
            1.9866,
            0.5851,
        ]
    ]

    netzwerk.schicht2.gewichte = [
        [1.2142, 0.8653, -0.2282, -0.2028, 0.9729, -1.3938, 0.0808, 0.2845],
        [-0.2687, 0.1175, 0.0131, -0.0079, -0.1701, 0.2546, 0.1974, 0.24],
        [0.2806, 0.7469, -0.114, 0.0514, 0.683, 0.4402, -0.0339, -0.2846],
        [0.4742, 0.5785, 0.0451, -0.1938, 0.2171, 0.6416, -0.2316, -0.058],
        [0.5731, 0.7834, -0.1861, 0.0798, 0.9531, -0.0173, -0.2048, 0.0684],
        [0.1354, 0.0085, -0.0252, 0.1742, 0.1655, -0.2682, 0.1864, -0.0308],
        [-2.6353, -2.2848, -0.1197, -0.2494, -1.8291, 1.2427, -0.1891, -0.2045],
        [0.7434, 0.4293, 0.171, -0.409, 0.9035, -0.0922, -0.2061, -0.2992],
    ]

    netzwerk.schicht2.bias = [
        [
            0.2191,
            0.1604,
            -0.0102,
            -0.4476,
            -0.0385,
            0.4184,
            -0.3466,
            0.3397,
        ]
    ]

    netzwerk.schicht3.gewichte = [
        [0.8571],
        [1.0345],
        [-0.0774],
        [-0.1742],
        [0.9437],
        [-0.9353],
        [-0.0455],
        [-0.2133],
    ]

    netzwerk.schicht3.bias = [[0.4596]]


optimal()
# geschichte = netzwerk.trainieren(eingaben, lösungen)
(ausgaben, _) = netzwerk.vorwärts_durchlauf(eingaben)
geschichte = [ausgaben]


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
