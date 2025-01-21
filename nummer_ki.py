import numpy as np
import matplotlib.pyplot as plt
import lade_daten
import os


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
        losses = -np.log(np.sum(abgeschnittene_prädiktionen * lösungen, axis=1))
        kost = np.mean(losses)
        return kost

    def ableiten_auf_prädiktionen(prädiktionen, lösungen):
        daten_größe = len(prädiktionen)
        dverluste_dausgaben = -lösungen / prädiktionen
        dkost_dausgaben = dverluste_dausgaben / daten_größe
        return dkost_dausgaben


class Netzwerk:
    def __init__(selbst):
        selbst.lern_rate = 0.01
        selbst.schicht1 = Schicht(784, 20)
        selbst.aktivierung1 = ReLU()
        selbst.schicht2 = Schicht(20, 10)
        selbst.aktivierung2 = Softmax()
        selbst.kost_funktion = Categorical_Cross_Entropy

    def vorwärts_durchlauf(selbst, eingaben):
        rohe_ausgaben_1 = selbst.schicht1.vorwärts(eingaben)
        aktivierte_ausgaben_1 = selbst.aktivierung1.vorwärts(rohe_ausgaben_1)
        rohe_ausgaben_2 = selbst.schicht2.vorwärts(aktivierte_ausgaben_1)
        aktivierte_ausgaben_2 = selbst.aktivierung2.vorwärts(rohe_ausgaben_2)
        return aktivierte_ausgaben_2

    def rückwärts_durchlauf(selbst, prädiktionen, lösungen):
        dkost_dprädiktionen = selbst.kost_funktion.ableiten_auf_prädiktionen(
            prädiktionen, lösungen
        )
        dkost_daktivierte_ausgaben_2 = selbst.aktivierung2.ableiten_auf_rohe_ausgaben(
            dkost_dprädiktionen
        )
        selbst.schicht2.rückwärts(dkost_daktivierte_ausgaben_2)
        dkost_daktivierte_ausgaben_1 = selbst.schicht2.deingaben
        dkost_drohe_ausgaben_1 = selbst.aktivierung1.ableiten_auf_rohe_ausgaben(
            dkost_daktivierte_ausgaben_1
        )

        selbst.schicht1.rückwärts(dkost_drohe_ausgaben_1)

    def SGD_optimierer(selbst):
        for schicht in [selbst.schicht1, selbst.schicht2]:
            schicht.gewichte -= schicht.dgewichte * selbst.lern_rate
            schicht.bias -= schicht.dbias * selbst.lern_rate

    def berechne_genauigkeit(selbst, prädiktionen, lösungen):
        lösungen = np.argmax(lösungen, axis=1)
        prädiktionen = np.argmax(prädiktionen, axis=1)
        vergleich = prädiktionen == lösungen
        genauigkeit = np.mean(vergleich)
        return genauigkeit

    def daten_vermischen(selbst, eingaben, lösungen):
        schlüssel = np.array(range(eingaben.shape[0]))
        np.random.shuffle(schlüssel)
        return (eingaben[schlüssel], lösungen[schlüssel])

    def vermischter_mini_batch(
        selbst, eingaben, lösungen, batch_größe, anzahl_trainings_daten
    ):
        eingaben, lösungen = selbst.daten_vermischen(eingaben, lösungen)
        mini_batches_eingaben = [
            (eingaben)[k : k + batch_größe]
            for k in range(0, anzahl_trainings_daten, batch_größe)
        ]
        mini_batches_lösungen = [
            (lösungen)[k : k + batch_größe]
            for k in range(0, anzahl_trainings_daten, batch_größe)
        ]
        return mini_batches_eingaben, mini_batches_lösungen

    def trainieren(selbst, eingaben, lösungen):
        batch_größe = 4
        epochen = 500
        anzahl_trainings_daten = len(eingaben)
        kosten = []
        genauigkeiten = []
        for epoche in range(epochen):
            mini_batches_eingaben, mini_batches_lösungen = (
                selbst.vermischter_mini_batch(
                    eingaben, lösungen, batch_größe, anzahl_trainings_daten
                )
            )
            for batch_eingaben, batch_lösungen in zip(
                mini_batches_eingaben, mini_batches_lösungen
            ):
                prädiktionen = selbst.vorwärts_durchlauf(batch_eingaben)
                selbst.rückwärts_durchlauf(prädiktionen, batch_lösungen)
                selbst.SGD_optimierer()
                kost = selbst.kost_funktion.berechnen(prädiktionen, batch_lösungen)
                kosten.append(kost)
                genauigkeit = selbst.berechne_genauigkeit(prädiktionen, batch_lösungen)
                genauigkeiten.append(genauigkeit)
            avg_kost = np.mean(kosten)
            print(
                "Epoche: ",
                epoche,
                " Genauigkeit: ",
                np.mean(genauigkeiten),
                "Kost: ",
                avg_kost,
            )
            if avg_kost <= 0.01:
                break


netzwerk = Netzwerk()


def trainieren(netzwerk):
    bilder, beschriftungen = lade_daten.lade_trainings_daten()
    netzwerk.trainieren(bilder, beschriftungen)


def interaktives_testen(netzwerk):
    bilder, _ = lade_daten.lade_test_daten()
    while True:
        index = int(input("Zahl von 0 bis 9999: "))
        prädiktion = netzwerk.vorwärts_durchlauf(bilder[index])
        prädiktion = np.argmax(prädiktion)
        plt.imshow(bilder[index].reshape(28, 28), cmap="Greys")
        plt.title(prädiktion)
        plt.show()


def testen(netzwerk):
    bilder, beschriftungen = lade_daten.lade_test_daten()
    prädiktionen = netzwerk.vorwärts_durchlauf(bilder)
    kost = netzwerk.kost_funktion.berechnen(prädiktionen, beschriftungen)
    genauigkeit = netzwerk.berechne_genauigkeit(prädiktionen, beschriftungen)
    print(
        "Genauigkeit: ",
        genauigkeit,
        "Kost: ",
        kost,
    )


soll_trainieren = True
interaktiv = False

if not os.path.isfile("netzwerk.pickle") or soll_trainieren == True:
    netzwerk = Netzwerk()
    trainieren(netzwerk)
    lade_daten.speicher_netzwerk(netzwerk)
else:
    netzwerk = lade_daten.lade_netzwerk()
    if interaktiv == True:
        interaktives_testen(netzwerk)
    else:
        testen(netzwerk)
