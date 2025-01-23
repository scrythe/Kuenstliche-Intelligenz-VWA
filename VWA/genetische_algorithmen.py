class Gegenstand:
    def __init__(selbst, name, gewicht, geldwert):
        selbst.name = name
        selbst.gewicht = gewicht
        selbst.geldwert = geldwert

gegenstände_liste = [
    Gegenstand("Handy", gewicht=3, geldwert=5),
    Gegenstand("Laptop", 6, 10),
    Gegenstand("Diamant", 1, 30),
    Gegenstand("Brot", 1, 1)
]

individuum = [0, 1, 0, 1]

######

import random

def erzeuge_zufällige_gegenstände_liste():
    zufällige_gegenstaende = []
    for _ in gegenstände_liste: # für jedes Element in Gegenstände Liste
        bit = random.choice([0, 1]) # zufällig 1 oder 0 wählen
        zufällige_gegenstaende.append(bit)
    return zufällige_gegenstaende

class Individuum:
    def __init__(selbst, gegenstände):
        selbst.gegenstände = gegenstände

population_groesse = 50
def erzeuge_initiale_population():
    population = []
    while population_groesse > len(population):
        zufaellige_gegenstaende = erzeuge_zufällige_gegenstände_liste()
        individuum = Individuum(zufaellige_gegenstaende)
        population.append(individuum)
    return population

#####

gewicht_limit = 10

class Individuum(Individuum):
    ...
    def fitness(selbst):
        gesamt_gewicht = 0
        gesamt_geldwert = 0
        # Gehe jeden Gegenstand des Individuum 
        for gegenstand, ist_verwendet in zip(gegenstände_liste, selbst.gegenstände):
            if ist_verwendet:
                gesamt_gewicht += gegenstand.gewicht
                gesamt_geldwert += gegenstand.geldwert
        if gesamt_gewicht > gewicht_limit:
            gesamt_geldwert = 0
        return gesamt_geldwert

#####

def tournament(gegner1, gegner2):
    if gegner1.fitness() > gegner2.fitness():
        return gegner1
    else:
        return gegner2

def selektion(population):
    gegner = random.sample(population, 4) # 4 zufällige Individuuen
    gewinner1 = tournament(gegner[0], gegner[1])
    gewinner2 = tournament(gegner[2], gegner[3])
    return [gewinner1, gewinner2]

print(selektion(erzeuge_initiale_population()))