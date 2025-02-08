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


class Individual:
    def __init__(self, mass_limit):
        self.create_random_item_bits()
        self.mass_limit = mass_limit

    def create_random_item_bits(self):
        self.item_bits = []
        for _ in item_list:  # für jedes Element in der Gegenstände Liste
            bit = random.choice([0, 1])  # zufällig 1 oder 0 wählen
            self.item_bits.append(bit)

    def calculate_fitness(self):
        total_mass = 0
        total_value = 0
        # Gehe jeden Gegenstand des Individuums durch
        for item, is_in_backpack in zip(item_list, self.item_bits):
            if is_in_backpack:
                total_mass += item.mass
                total_value += item.value

        if total_mass > self.mass_limit:
            self.fitness_value = 0
            return

        self.fitness_value = total_value

    def print_items_in_backpack(self):
        for item, is_in_backpack in zip(item_list, self.item_bits):
            if is_in_backpack:
                print(
                    f"Gegenstand: {item.name}, Masse: {item.mass}kg, Geldwert: {item.value}€"
                )


REPRODUCTION_RATE = 0.6
CROSS_OVER_RATE = 0.2
MUTATION_RATE = 0.1


class Population:
    def __init__(self, population_size, mass_limit):
        self.population_size = population_size
        self.create_initial_population(mass_limit)

    def calculate_fitness(self):
        for individual in self.population:
            individual.calculate_fitness()

    def create_initial_population(self, mass_limit):
        self.population = []
        while population_size > len(self.population):
            individual = Individual(mass_limit)
            self.population.append(individual)

    def print_population(self):
        for individual in self.population:
            print(individual.item_bits)

    def create_new_population(self):
        new_population = []
        best_individual = copy.deepcopy(self.population[0])
        new_population.append(best_individual)
        while population_size > len(new_population):
            parent1, parent2 = selection(self.population)
            if CROSS_OVER_RATE > random.random():
                crossover(parent1, parent2)

            if MUTATION_RATE > random.random():
                mutation(parent1)
            if MUTATION_RATE > random.random():
                mutation(parent2)

            new_population.append(parent1)
            new_population.append(parent2)

        return new_population

    def start(self):
        self.calculate_fitness()
        for _ in range(500):
            self.population = self.create_new_population()
            self.calculate_fitness()

            self.population.sort(
                reverse=True, key=lambda individual: individual.fitness_value
            )

            best_individual = self.population[0]
            print(best_individual.fitness_value)


population_size = 50
mass_limit = 10


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


def crossover(parent1, parent2):
    bits_amount = len(parent1.item_bits)
    half_amount = int(bits_amount / 2)

    # Erste Hälfte von Elternteil 1 plus zweite Hälfte von Elternteil 2
    temp1_bits = parent1.item_bits[:half_amount] + parent2.item_bits[half_amount:]

    # Erste Hälfte von Elternteil 2 plus zweite Hälfte von Elternteil 1
    temp2_bits = parent2.item_bits[:half_amount] + parent1.item_bits[half_amount:]

    parent1.item_bits = temp1_bits
    parent2.item_bits = temp2_bits


def mutation(individual):
    bits_amount = len(individual.item_bits)
    random_bit = random.randrange(bits_amount)
    individual.item_bits[random_bit] = (
        1 - individual.item_bits[random_bit]
    )  # Wenn 1, wird zu 0 und umgekehrt


Population(population_size, mass_limit).start()
