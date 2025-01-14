import numpy as np
import pathlib
import matplotlib.pyplot as plt

# from data import get_mnist
import math


def hole_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/daten/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


class Schicht:
    def __init__(selbst, anzahl_eingaben, anzahl_neuronen):
        selbst.gewichte = np.random.rand(anzahl_eingaben, anzahl_neuronen)
        selbst.bias = np.zeros((1, anzahl_neuronen))

    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.dot(eingaben, selbst.gewichte) + selbst.bias


class Sigmoid:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = 1 / (1 + np.exp(-eingaben))


class ReLU:
    def vorwärts(selbst, eingaben):
        selbst.ausgaben = np.maximum(0, eingaben)


class SoftMax:
    def vorwärts(selbst, eingaben):
        exponierte_werte = np.exp(eingaben - np.max(eingaben, axis=1, keepdims=True))
        normalisierte_basis = np.sum(exponierte_werte, axis=1, keepdims=True)
        selbst.ausgaben = exponierte_werte / normalisierte_basis


# class Categorical_Cross_Entropy:
#     def calculate(selbst, ausgaben, lösungen):
#         ausgaben_indexe = [range(len(ausgaben))]
#         gewählte_ausgaben = np.sum(ausgaben * lösungen, axis=1)
#         losses = -np.log(gewählte_ausgaben)
#         avg_loss = np.mean(losses)
#         return avg_loss


# schicht1 = Schicht(1, 2)
# aktivierung1 = ReLU()

# schicht2 = Schicht(2, 2)
# aktivierung2 = SoftMax()

# loss_function = Categorical_Cross_Entropy()

# schicht1.forward([[2], [1]])
# aktivierung1.forward(schicht1.ausgaben)

# schicht2.forward(aktivierung1.ausgaben)
# aktivierung2.forward(schicht2.ausgaben)

# # loss = -math.log(aktivierung2.ausgabe[0])
# loss_value = loss_function.calculate(np.array([[0.1, 0.5], [0.1, 1]]), [[0, 1], [1, 0]])
# print(loss_value)


class Netzwerk:
    def __init__(selbst):
        selbst.schicht1 = Schicht(784, 20)
        selbst.aktivierung1 = ReLU()

        selbst.schicht2 = Schicht(20, 10)
        selbst.aktivierung2 = SoftMax()

    def vorwärtspropagierung(selbst, eingaben):
        selbst.schicht1.vorwärts(eingaben)
        selbst.aktivierung1.vorwärts(selbst.schicht1.ausgaben)

        selbst.schicht2.vorwärts(selbst.aktivierung1.ausgaben)
        selbst.aktivierung2.vorwärts(selbst.schicht2.ausgaben)

        ergebnisse = np.argmax(selbst.aktivierung2.ausgaben, axis=1)
        return ergebnisse


bilder, beschriftungen = hole_mnist()
net = Netzwerk()

index = int(input("Zahl von 0 bis 59999: "))
bild = bilder[index]
plt.imshow(bild.reshape(28, 28), cmap="Greys")
ergebnisse = net.vorwärtspropagierung(bilder[index])
plt.title(ergebnisse[0])
plt.show()
