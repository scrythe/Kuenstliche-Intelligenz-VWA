import matplotlib.pyplot as plt

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.erstelle_netzwerk import zeichne_netzwerk
from src.hilfsfunktionen import erstelle_bild

grosse = 6
y_ratio = 0.5
zeichne_netzwerk(grosse, y_ratio, [3, 4, 4, 2])
plt.show()
