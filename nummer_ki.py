import numpy as np
import matplotlib.pyplot as plt
import lade_daten


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

    def ableiten_auf_rohe_ausgaben(selbst, dkost_daktivierte_ausgaben):
        return dkost_daktivierte_ausgaben * (selbst.eingaben > 0)


class Softmax:
    def vorwärts(selbst, eingaben):
        selbst.eingaben = eingaben
        max_wert = np.max(eingaben, axis=1, keepdims=True)
        exponierte_werte = np.exp(eingaben - max_wert)
        normalisierte_basis = np.sum(exponierte_werte, axis=1, keepdims=True)
        selbst.ausgaben = exponierte_werte / normalisierte_basis
        return selbst.ausgaben

    def ableiten_auf_rohe_ausgaben(selbst, dkost_daktivierte_ausgaben):
        deingaben = np.empty_like(dkost_daktivierte_ausgaben)
        for index, (einzelne_ausgabe, einzelne_dkost_daktivierte_ausgabe) in enumerate(
            zip(selbst.ausgaben, dkost_daktivierte_ausgaben)
        ):
            einzelne_ausgabe = einzelne_ausgabe.reshape(-1, 1)
            jacobi_matrix = np.diagflat(einzelne_ausgabe) - np.dot(
                einzelne_ausgabe, einzelne_ausgabe.T
            )
            deingaben[index] = np.dot(jacobi_matrix, einzelne_dkost_daktivierte_ausgabe)
        return deingaben


class Categorical_Cross_Entropy:
    def berechnen(prädiktionen, lösungen):
        abgeschnittene_prädiktionen = np.clip(prädiktionen, 1e-7, 1 - 1e-7)
        prädiktionen_indexe = range(len(prädiktionen))
        gewählte_prädiktionen = abgeschnittene_prädiktionen[
            prädiktionen_indexe, lösungen
        ]
        losses = -np.log(gewählte_prädiktionen)
        kost = np.mean(losses)
        return kost

    def ableiten_auf_prädiktionen(prädiktionen, lösungen):
        daten_größe = len(prädiktionen)
        anzahl_verschiedener_lösungen = len(prädiktionen[0])
        lösungen = np.eye(anzahl_verschiedener_lösungen)[lösungen]
        dverluste_dausgaben = -lösungen / prädiktionen
        dkost_dausgaben = dverluste_dausgaben / daten_größe
        return dkost_dausgaben


class Netzwerk:
    def __init__(selbst):
        selbst.lern_rate = 0.01
        selbst.schicht1 = Schicht(784, 20)
        selbst.aktivierung1 = ReLU()
        selbst.schicht2 = Schicht(20, 20)
        selbst.aktivierung2 = ReLU()
        selbst.schicht3 = Schicht(20, 10)
        selbst.aktivierung3 = Softmax()
        selbst.kost_funktion = Categorical_Cross_Entropy

    def vorwärts_durchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        rohe_ausgaben_3 = selbst.schicht3.vorwärts(aktivierte_ausgaben_2)
        aktivierte_ausgaben_3 = selbst.aktivierung3.vorwärts(rohe_ausgaben_3)
        return aktivierte_ausgaben_3

    def rückwärts_durchlauf(selbst, prädiktionen, lösungen):
        dkost_dprädiktionen = selbst.kost_funktion.ableiten_auf_prädiktionen(
            prädiktionen, lösungen
        )
        dkost_daktivierte_ausgaben_3 = selbst.aktivierung3.ableiten_auf_rohe_ausgaben(
            dkost_dprädiktionen
        )
        selbst.schicht3.rückwärts(dkost_daktivierte_ausgaben_3)
        dkost_daktivierte_ausgaben_2 = selbst.schicht3.deingaben
        dkost_drohe_ausgaben_2 = selbst.aktivierung2.ableiten_auf_rohe_ausgaben(
            dkost_daktivierte_ausgaben_2
        )

        selbst.schicht2.rückwärts(dkost_drohe_ausgaben_2)
        dkost_daktivierte_ausgaben_2 = selbst.schicht2.deingaben
        dkost_drohe_ausgaben_2 = selbst.aktivierung2.ableiten_auf_rohe_ausgaben(
            dkost_daktivierte_ausgaben_2
        )

        selbst.schicht1.rückwärts(dkost_drohe_ausgaben_2)

    def SGD_optimierer(selbst):
        for schicht in [selbst.schicht1, selbst.schicht2, selbst.schicht3]:
            schicht.gewichte -= schicht.dgewichte * selbst.lern_rate
            schicht.bias -= schicht.dbias * selbst.lern_rate

    def berechne_genauigkeit(selbst, prädiktionen, lösungen):
        prädiktionen = np.argmax(prädiktionen, axis=1)
        vergleich = prädiktionen == lösungen
        genauigkeit = np.mean(vergleich)
        return genauigkeit

    def trainieren(selbst, eingaben, lösungen):
        batch_größe = 128
        epochen = 20
        anzahl_trainings_daten = len(eingaben)
        kost = 0
        genauigkeit = 0
        mini_batches_eingaben = [
            (eingaben)[k : k + batch_größe]
            for k in range(0, anzahl_trainings_daten, batch_größe)
        ]
        mini_batches_lösungen = [
            (lösungen)[k : k + batch_größe]
            for k in range(0, anzahl_trainings_daten, batch_größe)
        ]
        for _ in range(epochen):
            for batch_eingaben, batch_lösungen in zip(
                mini_batches_eingaben, mini_batches_lösungen
            ):
                prädiktionen = selbst.vorwärts_durchlauf(batch_eingaben)
                selbst.rückwärts_durchlauf(prädiktionen, batch_lösungen)
                selbst.SGD_optimierer()
                kost = selbst.kost_funktion.berechnen(prädiktionen, batch_lösungen)
                genauigkeit = selbst.berechne_genauigkeit(prädiktionen, batch_lösungen)
            print(
                "Genauigkeit: ",
                genauigkeit,
                "Kost: ",
                kost,
            )


netzwerk = Netzwerk()


def trainieren():
    bilder, beschriftungen = lade_daten.lade_trainings_daten()
    keys = np.array(range(bilder.shape[0]))
    np.random.shuffle(keys)
    bilder = bilder[keys]
    beschriftungen = beschriftungen[keys]
    bilder = (
        bilder.reshape(bilder.shape[0], -1).astype(np.float32) - 127.5
    ) / 127.5  # Zwischen -1 und 1
    netzwerk.trainieren(bilder, beschriftungen)


def testen():
    bilder, _ = lade_daten.lade_test_daten()
    bilder = (
        bilder.reshape(bilder.shape[0], -1).astype(np.float32) - 127.5
    ) / 127.5  # Zwischen -1 und 1
    while True:
        index = int(input("Zahl von 0 bis 9999: "))
        prädiktion = netzwerk.vorwärts_durchlauf(bilder[index])
        prädiktion = np.argmax(prädiktion)
        plt.imshow(bilder[index].reshape(28, 28), cmap="Greys")
        plt.title(prädiktion)
        plt.show()


trainieren()
testen()
