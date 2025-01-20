import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def f(x):
    return 0.05 * np.power(x, 3) - 0.5 * x


def Mean_Squared_Error(prädiktionen, lösungen):
    verluste = np.power(prädiktionen - lösungen, 2)
    kost = np.mean(verluste)
    return kost


class Netzwerk:
    def __init__(selbst):
        selbst.a = 0.01
        selbst.b = 0.01

    def vorwärts_durchlauf(selbst, eingaben):
        prädiktionen = selbst.a * np.power(eingaben, 3) + selbst.b * eingaben
        return prädiktionen

    def rückwärts_durchlauf(selbst, eingaben, lösungen):
        prädiktionen = selbst.vorwärts_durchlauf(eingaben)
        dverlust_dprädiktion = 2 * (prädiktionen - lösungen)
        dkosten_dprädiktion = dverlust_dprädiktion / len(prädiktionen)

        dkosten_da = dkosten_dprädiktion = dkosten_dprädiktion * np.power(eingaben, 3)
        dkosten_db = dkosten_dprädiktion = dkosten_dprädiktion * eingaben
        da = np.sum(dkosten_da)
        db = np.sum(dkosten_db)
        gradiant = (da, db)
        return gradiant

    def parameter_anpassen(selbst, gradiant):
        lern_rate = 0.0001
        selbst.a -= gradiant[0] * lern_rate
        selbst.b -= gradiant[1] * lern_rate

    def trainieren(selbst, eingaben, lösungen):
        geschichte = []
        kost = 1
        while kost > 0.0001:
            prädiktionen = selbst.vorwärts_durchlauf(eingaben)
            kost = Mean_Squared_Error(prädiktionen, lösungen)
            geschichte.append(prädiktionen)
            gradiant = selbst.rückwärts_durchlauf(eingaben, lösungen)
            selbst.parameter_anpassen(gradiant)
        print(selbst.a)
        print(selbst.b)
        return geschichte


eingaben = np.arange(-5, 5, 0.5).T
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
    return (linie,)


def update(epoche):
    linie.set_ydata(geschichte[epoche])
    punkte.set_ydata(geschichte[epoche])
    ax.set_title(f"Epoche {epoche}")
    return (linie,)


framge_range = range(0, len(geschichte), 10)

ani = animation.FuncAnimation(
    fig, update, frames=framge_range, init_func=init, interval=100, repeat=False
)

plt.show()
