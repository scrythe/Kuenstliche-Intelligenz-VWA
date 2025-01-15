import numpy as np
import pathlib
import matplotlib.pyplot as plt


def hole_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/daten/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen, aktivierungs_funktion):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))
        selbst.aktivierungs_funktion = aktivierungs_funktion

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias
        selbst.aktivierte_ausgaben = selbst.aktivierungs_funktion(selbst.ausgaben)


def Sigmoid(eingaben):
    return 1 / (1 + np.exp(-eingaben))


def ReLU(eingaben):
    return np.maximum(0, eingaben)


def Softmax(eingaben):
    exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
    normalisierte_basis = np.sum(exponierte_werte, axis=1, keepdims=True)
    return exponierte_werte / normalisierte_basis


class Categorical_Cross_Entropy:
    def calculate(selbst, ausgaben, lösungen):
        ausgaben_indexe = [range(len(ausgaben))]
        gewählte_ausgaben = np.sum(ausgaben * lösungen, axis=1)
        losses = -np.log(gewählte_ausgaben)
        avg_loss = np.mean(losses)
        return avg_loss


class Netzwerk:
    def __init__(selbst):
        anzahl_eingabe_neuronen = 784
        anzahl_versteckte_neuronen = 20
        anzahl_ausgabe_neuronen = 10
        selbst.versteckte_schicht = Schicht(
            anzahl_eingabe_neuronen, anzahl_versteckte_neuronen, ReLU
        )
        selbst.ausgabe_schicht = Schicht(
            anzahl_versteckte_neuronen, anzahl_ausgabe_neuronen, Softmax
        )

    def vorwärtspropagierung(selbst, eingaben):
        selbst.versteckte_schicht.vorwärts(eingaben)
        selbst.ausgabe_schicht.vorwärts(selbst.versteckte_schicht.ausgaben)
        vorhergesagte_werte = np.argmax(selbst.ausgabe_schicht.ausgaben, axis=1)
        return vorhergesagte_werte


bilder, beschriftungen = hole_mnist()
netzwerk = Netzwerk()
index = int(input("Zahl von 0 bis 59999: "))
bild = bilder[index]
plt.imshow(bild.reshape(28, 28), cmap="Greys")
vorhergesagte_werte = netzwerk.vorwärtspropagierung(bilder[index])
plt.title(vorhergesagte_werte[0])
plt.show()
