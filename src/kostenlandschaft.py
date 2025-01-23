import numpy as np
from verlustfunktionen import MittlererQuadratischerFehler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


def f(x):
    return 2 * x + 2


eingaben = np.arange(-3, 3, 0.1)
ziele = f(eingaben)


def berechne_kostenfunktion_auf_gewicht_bias(gewichte, bias):
    vorhersagen = gewichte * eingaben + bias
    verlust = MittlererQuadratischerFehler.verlust(vorhersagen, ziele)
    return verlust


def berechne_gradient(gewicht, bias):
    vorhersagen = gewicht * eingaben + bias
    verlust_gradient = MittlererQuadratischerFehler.rueckwaerts(vorhersagen, ziele)

    gradient_gewicht = np.dot(eingaben.T, verlust_gradient)
    gradient_bias = np.sum(verlust_gradient)
    return [gradient_gewicht, gradient_bias]


def zeichne_landschaft():
    gewichtsbereich = np.arange(-1, 4, 0.1)
    biasbereich = np.arange(-1, 4, 0.1)
    gewichte, bias = np.meshgrid(gewichtsbereich, biasbereich)
    verluste = np.array(
        [
            [
                berechne_kostenfunktion_auf_gewicht_bias(gewicht, bias_wert)
                for gewicht in gewichtsbereich
            ]
            for bias_wert in biasbereich
        ]
    )

    ax.plot_surface(
        gewichte,
        bias,
        verluste,
        cmap="coolwarm",
    )  # Kostenlandschaft


def zeichne_ball_und_gradient():
    ball_gewicht = -1
    ball_bias = -1
    verlust = berechne_kostenfunktion_auf_gewicht_bias(ball_gewicht, ball_bias)
    ax.plot(
        ball_gewicht,
        ball_bias,
        [verlust],
        "ro",
    )  # Roter Ball

    # gradient = berechne_gradient(ball_gewicht, ball_bias)
    # verlust2 = berechne_gradient(ball_gewicht - gradient[0], ball_bias - gradient[1])

    # # ax.quiver(
    # #     ball_gewicht,
    # #     ball_bias,
    # #     verlust,
    # #     -gradient[0],
    # #     -gradient[1],
    # #     verlust2 - verlust,
    # #     color="black",
    # #     linewidth=2,
    # #     label="Gradient",
    # # )


fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection="3d", computed_zorder=False)

zeichne_landschaft()
zeichne_ball_und_gradient()

# Achsentitel
ax.set_title("Kostenlandschaft in Bezug auf Gewichte und Bias")
ax.set_xlabel("Gewicht (w)")
ax.set_ylabel("Bias (b)")
ax.invert_yaxis()
ax.set_zlabel("Verlust (L)")


plt.show()
