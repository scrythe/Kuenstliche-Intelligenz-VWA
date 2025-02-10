import random
import copy


class Item:
    def __init__(self, name, mass, value):
        self.name = name
        self.mass = mass
        self.value = value


item_list = [
    # Handy mit 3kg Masse und einem Geldwert von 5€
    Item("Handy", mass=3, value=5),
    Item("Laptop", 6, 10),
    Item("Diamant", 1, 30),
    Item("Brot", 1, 1),
]


def generate_item_list(length):
    items = []
    for i in range(length):
        items.append(Item(i, i, i))
    return items


item_list = generate_item_list(10)


class Individual:
    def __init__(self, item_bits):
        self.item_bits = item_bits

    @staticmethod
    def create_random_individual():
        random_item_bits = []
        for _ in item_list:  # für jedes Element in der Gegenstände Liste
            bit = random.choice([0, 1])  # zufällig 1 oder 0 wählen
            random_item_bits.append(bit)
        return Individual(random_item_bits)

    def calculate_fitness(self, mass_limit):
        total_mass = 0
        total_value = 0
        # Gehe jeden Gegenstand des Individuums durch
        for item, is_in_backpack in zip(item_list, self.item_bits):
            if is_in_backpack:
                total_mass += item.mass
                total_value += item.value

        if total_mass > mass_limit:
            self.fitness_value = 0
            return

        self.fitness_value = total_value

    def print_items_in_backpack(self):
        for item, is_in_backpack in zip(item_list, self.item_bits):
            if is_in_backpack:
                print(
                    f"Gegenstand: {item.name}, Masse: {item.mass}kg, Geldwert: {item.value}€"
                )


class Population:
    def __init__(self, population_size):
        self.population_size = population_size
        self.create_initial_population()

    def calculate_fitness(self, mass_limit):
        for individual in self.population:
            individual.calculate_fitness(mass_limit)

    def create_initial_population(self):
        self.population = []
        while population_size > len(self.population):
            individual = Individual.create_random_individual()
            self.population.append(individual)

    def print_population(self):
        for individual in self.population:
            print(individual.item_bits)

    def create_new_population(self):
        new_population = []
        best_individuals = self.population[0:2]
        new_population.extend(best_individuals)
        while population_size > len(new_population):
            parent1, parent2 = selection(self.population)

            child1, child2 = crossover_parents(parent1, parent2)

            mutatate_child(child1)
            mutatate_child(child2)

            new_population.append(child1)
            new_population.append(child2)

        return new_population

    def start(self, mass_limit):
        self.calculate_fitness(mass_limit)
        for _ in range(500):
            self.population = self.create_new_population()
            self.calculate_fitness(mass_limit)

            self.population.sort(
                reverse=True, key=lambda individual: individual.fitness_value
            )

            best_individual = self.population[0]

        print(best_individual.fitness_value)
        print(best_individual.item_bits)


population_size = 20
mass_limit = 3000


def tournament(enemy1, enemy2):
    if enemy1.fitness_value > enemy2.fitness_value:
        return enemy1
    else:
        return enemy2


def selection(population):
    enemies = random.sample(population, 4)  # 4 zufällige Individuuen
    winner1 = tournament(enemies[0], enemies[1])
    winner2 = tournament(enemies[2], enemies[3])
    return [winner1, winner2]


def crossover_parents(parent1, parent2):
    bits_amount = len(parent1.item_bits)
    half_amount = int(bits_amount / 2)

    # Erste Hälfte von Elternteil 1 plus zweite Hälfte von Elternteil 2
    child1_bits = parent1.item_bits[:half_amount] + parent2.item_bits[half_amount:]

    # Erste Hälfte von Elternteil 2 plus zweite Hälfte von Elternteil 1
    child2_bits = parent2.item_bits[:half_amount] + parent1.item_bits[half_amount:]

    child1 = Individual(child1_bits)
    child2 = Individual(child2_bits)
    return (child1, child2)


def mutatate_child(individual):
    bits_amount = len(individual.item_bits)
    random_bit = random.randrange(bits_amount)
    individual.item_bits[random_bit] = (
        1 - individual.item_bits[random_bit]
    )  # 1 wird zu null un umgekehrt


Population(population_size).start(mass_limit)
