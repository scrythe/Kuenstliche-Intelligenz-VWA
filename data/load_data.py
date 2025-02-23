import gzip
import numpy as np
import pickle
from neuronales_netzwerk.netzwerk import Netzwerk


def load_images(datei):
    with gzip.open(datei, "r") as f:
        f.read(4)  # Überspringen des Headers (Magic Number)
        anzahl_bilder = int.from_bytes(f.read(4), "big")
        f.read(8)  # Überspringen des Headers (Anzahl Reihen und Zeilen)

        # Lesen der Bilddaten
        # Pixelwerte sind von 0 bis 255 als unsigned Byte gespeichert
        bilder_daten = f.read()
        bilder = np.frombuffer(bilder_daten, dtype=np.uint8).reshape(784, anzahl_bilder)
        bilder = (
            bilder.reshape(bilder.shape[0], -1).astype(np.float32) - 127.5
        ) / 127.5  # Zwischen -1 und 1
        return bilder


def load_labels(datei):
    with gzip.open(datei, "r") as f:
        f.read(8)  # Überspringen des Headers (Magic Number und Anzahl der Labels)
        beschriftungs_daten = f.read()
        beschriftungen = np.frombuffer(beschriftungs_daten, dtype=np.uint8)
        beschriftungen = np.eye(10)[beschriftungen].T
        return beschriftungen


def load_trainings_data():
    bilder = load_images("data/train-images-idx3-ubyte.gz")
    beschriftungen = load_labels("data/train-labels-idx1-ubyte.gz")
    return bilder, beschriftungen


def load_test_data():
    bilder = load_images("data/t10k-images-idx3-ubyte.gz")
    beschriftungen = load_labels("data/t10k-labels-idx1-ubyte.gz")
    return bilder, beschriftungen


def save_network(netzwerk: Netzwerk):
    with open("network.pickle", "wb") as f:
        pickle.dump(netzwerk, f)
    print("fertig")


def load_network() -> Netzwerk:
    with open("network.pickle", "rb") as f:
        netzwerk = pickle.load(f)
    return netzwerk
