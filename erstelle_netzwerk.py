# Zeichne Netzwerk

import matplotlib.pyplot as plt
from io import BytesIO


def neuron_x_pos(schicht, radius, horizont_abstand):
    x_links_offset = radius + horizont_abstand / 6
    x_position = schicht * horizont_abstand + x_links_offset
    return x_position


def neuron_y_pos(neuron, anzahl, vertikal_abstand, y_laenge, y_text_offset):
    y_oben_offset = (
        vertikal_abstand * (anzahl - 1) / 2 + (y_laenge) / 2 + y_text_offset / 2
    )
    y_position = y_oben_offset - neuron * vertikal_abstand
    return y_position


def zeichne_neuronen(
    ax, schichten, radius, horizont_abstand, vertikal_abstand, y_laenge, y_text_offset
):
    for schicht, anzahl in enumerate(schichten):
        x_position = neuron_x_pos(schicht, radius, horizont_abstand)
        for neuron in range(anzahl):
            y_position = neuron_y_pos(
                neuron, anzahl, vertikal_abstand, y_laenge, y_text_offset
            )
            kreis = plt.Circle(
                (x_position, y_position),
                radius,
                color="w",
                ec="k",
                zorder=4,
            )
            ax.add_artist(kreis)


def zeichne_text(ax, schichten, radius, horizont_abstand, y_text_offset):
    for schicht, _ in enumerate(schichten):
        x_position = neuron_x_pos(schicht, radius, horizont_abstand)
        if schicht == 0:
            name = "Eingabe\nschicht"
        elif schicht == len(schichten) - 1:
            name = "Ausgabe\nschicht"
        else:
            name = f"versteckte\nSchicht {schicht}"
        ax.text(
            x_position,
            y_text_offset,
            name,
            ha="center",
            fontsize=10,
            color="black",
        )


def zeichne_gewichte(
    ax, schichten, radius, horizont_abstand, vertikal_abstand, y_laenge, y_text_offset
):
    for schicht_links, (anzahl1, anzahl2) in enumerate(
        zip(schichten[:-1], schichten[1:])
    ):
        schicht_rechts = schicht_links + 1
        x1 = neuron_x_pos(schicht_links, radius, horizont_abstand)
        x2 = neuron_x_pos(schicht_rechts, radius, horizont_abstand)
        for neuron1 in range(anzahl1):
            y1 = neuron_y_pos(
                neuron1, anzahl1, vertikal_abstand, y_laenge, y_text_offset
            )
            for neuron2 in range(anzahl2):
                y2 = neuron_y_pos(
                    neuron2, anzahl2, vertikal_abstand, y_laenge, y_text_offset
                )
                line = plt.Line2D(
                    [x1, x2],
                    [y1, y2],
                    c="k",
                )
                ax.add_artist(line)


def zeichne_netzwerk(groesse, y_ratio, schichten):
    fig = plt.figure(figsize=(groesse, groesse * y_ratio))
    # ax = fig.gca()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.set_ylim(0, y_ratio)
    ax.invert_yaxis()

    x_laenge = 1
    y_laenge = x_laenge * y_ratio
    radius = 0.03

    y_text_offset = 0.05
    x_rechts_offset = 2 * radius
    y_oben_offset = radius * 2 + y_text_offset + 0.05
    horizont_abstand = (x_laenge - x_rechts_offset) / (len(schichten) - 0.75)
    vertikal_abstand = (y_laenge - y_oben_offset) / (max(schichten) - 1)
    zeichne_neuronen(
        ax,
        schichten,
        radius,
        horizont_abstand,
        vertikal_abstand,
        y_laenge,
        y_text_offset,
    )
    zeichne_text(ax, schichten, radius, horizont_abstand, y_text_offset)
    zeichne_gewichte(
        ax,
        schichten,
        radius,
        horizont_abstand,
        vertikal_abstand,
        y_laenge,
        y_text_offset,
    )
    bild = BytesIO()
    plt.savefig(bild, format="png")
    bild.seek(0)
    plt.close()
    return bild
