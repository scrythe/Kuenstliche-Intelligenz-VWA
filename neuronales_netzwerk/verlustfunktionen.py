import numpy as np
from numpy import ndarray


class VerlustFunktion:
    """
    Template-Verlustfunktion.
    """

    @staticmethod
    def kosten(vorhersagen: ndarray, ziele: ndarray) -> float:
        """
        Berechnet den Mittelwert der Verluste.

        Parameter:
            vorhersagen (ndarray): Matrix der Vorhersagen (z.B. Ausgaben von Softmax).
            ziele (ndarray): Korrekte Lösungen.

        Rückgabewert:
            float: Der Mittellwert der Verluste.
        """
        raise NotImplementedError

    @staticmethod
    def rueckwaerts(vorhersagen: ndarray, ziele: ndarray) -> ndarray:
        """
        Berechnet den Gradienten der Verlustfunktion

        Parameter:
            vorhersagen (ndarray): Matrix der Vorhersagen (z.B. Ausgaben von Softmax).
            ziele (ndarray): Korrekte Lösungen.

        Rückgabewert:
            ndarray: Der berechnete Gradient für die Verlustfunktion (Veränderung der Vorhersagen auf den Verlust) (dL/da).
        """
        raise NotImplementedError


class Kreuzentropie(VerlustFunktion):
    """
    Kreuzentropie-Verlustfunktion.
    """

    @staticmethod
    def kosten(vorhersagen: ndarray, ziele: ndarray) -> float:
        vorhersagen = np.clip(vorhersagen, 1e-7, 1 - 1e-7)
        verluste = np.sum(ziele * np.log(vorhersagen), axis=1)
        kosten = -np.mean(verluste)
        return kosten

    @staticmethod
    def rueckwaerts(vorhersagen: ndarray, ziele: ndarray) -> ndarray:
        """
        Berechnet den Gradienten der Verlustfunktion in Kombination mit Softmax

        Parameter:
            vorhersagen (ndarray): Matrix der Vorhersagen (z.B. Ausgaben von Softmax).
            ziele (ndarray): Korrekte Lösungen.

        Rückgabewert:
            ndarray: Der berechnete Gradient für die Verlustfunktion in Kombination mit Softmax (Veränderung der rohen Ausgaben der aktuellen Schicht auf den Verlust) (dL/dz).
        """
        return (vorhersagen - ziele) / len(vorhersagen)


class MittlererQuadratischerFehler(VerlustFunktion):
    """
    Mean-Squared-Error (MSE)-Verlustfunktion.
    """

    @staticmethod
    def kosten(vorhersagen: ndarray, ziele: ndarray) -> float:
        # Berechnet den mittleren quadratischen Fehler (MSE).
        verluste = np.square(vorhersagen - ziele)
        kosten = np.mean(verluste)
        return kosten

    @staticmethod
    def rueckwaerts(vorhersagen: ndarray, ziele: ndarray) -> ndarray:
        # Berechnet den Gradienten der MSE-Verlustfunktion.
        return 2 * (vorhersagen - ziele) / len(vorhersagen)
