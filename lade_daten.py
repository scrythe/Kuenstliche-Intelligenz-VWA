import gzip
import numpy as np
import pickle


def lade_bilder(datei):
    with gzip.open(datei, "r") as f:
        f.read(4)  # Überspringen des Headers (Magic Number)
        anzahl_bilder = int.from_bytes(f.read(4), "big")
        f.read(8)  # Überspringen des Headers (Anzahl Reihen und Zeilen)

        # Lesen der Bilddaten
        # Pixelwerte sind von 0 bis 255 als unsigned Byte gespeichert
        bilder_daten = f.read()
        bilder = np.frombuffer(bilder_daten, dtype=np.uint8).reshape(anzahl_bilder, 784)
        bilder = (
            bilder.reshape(bilder.shape[0], -1).astype(np.float32) - 127.5)
        # .reshape(
        #     (anzahl_bilder, 28, 28)
        # )
        return bilder


def lade_beschriftungen(datei):
    with gzip.open(datei, "r") as f:
        f.read(8)  # Überspringen des Headers (Magic Number und Anzahl der Labels)
        beschriftungs_daten = f.read()
        beschriftungen = np.frombuffer(beschriftungs_daten, dtype=np.uint8)
        return beschriftungen


def lade_trainings_daten():
    bilder = lade_bilder("daten/train-images-idx3-ubyte.gz")
    beschriftung = lade_beschriftungen("daten/train-labels-idx1-ubyte.gz")
    return bilder, beschriftung


def lade_test_daten():
    bilder = lade_bilder("daten/t10k-images-idx3-ubyte.gz")
    beschriftung = lade_beschriftungen("daten/t10k-labels-idx1-ubyte.gz")
    return bilder, beschriftung

def speicher_netzwerk(netzwerk):
    with open("netzwerk.pickle", "wb") as f:
        pickle.dump(netzwerk, f)
    print("fertig")


def lade_netzwerk():
    with open("netzwerk.pickle", "rb") as f:
        netzwerk=pickle.load(f)
    return netzwerk
