
import tensorflow as tf

import numpy as np
from numpy import random as r
from numpy.random import randint
from copy import deepcopy

import random as random
import matplotlib.pyplot as plt

import math

from nn import Trainer
import seaborn as sns

class Chromosome(object):

    """Initializes a random solution with `n_features` features.
    """
    def __init__(self, n_features) :
        self.n_features = n_features

        self.vector = np.array([random.randint(0, 1) for i in range(n_features)])

        self.accuracy = 0.0
        self.fitness = 0.0

    """Inverts chromosome for one feature. Switches one feature from on to off
    or from off to on.
    """
    def mutate(self, iters=1) :
        for i in range(iters):
            mutation_index = randint(0, self.length() - 1)
            current = self.vector[mutation_index]
            self.vector[mutation_index] = 0 if current == 1 else 1

    """Returns the length of the solution, equal to the number of total features.
    """
    def length(self) :
        return len(self.vector)

    """Replaces the range (start, end) of one chromosome with the same range of
    the other chromosome.
    """
    def replace_range(self, other, start, end) :
        np.put(self.vector, range(start, end), other.vector[start:end])

class GA(object):

    """Genetic algorithm wrapper object handling feature selection on the neural
    network.

    Attributes:
        population: list of n (pop_size) randomly generated chromosomes
        nn: instantiated Trainer object with training and testing data loaded
        n_gens: maximum number of generations the algorithm will undergo if
                convergence is not detected (default=50)
        gen_step: parameter used for internal testing, splitting one run into
                  multiple sessions (default=10)
        pop_size: the size of the solution population (default=20)
        cross_rate: probability that two parents will perform crossover to
                    produce a child (default=0.9)
                Note: mutation rate is 1-cross_rate (default=0.1)
        pool_size: the size of the tournament pool used during selection
                   (default=2)
        elitism_length: the number of highly fit chromosomes that will
                        automatically move to the next generation (default=2)
        history: a record of the best solutions from generation to generation
        present_features: the features that are currently used in the population
    """
    def __init__(self, nn, n_gens=50, gen_step=10, pop_size=20, cross_rate=0.9,
                 pool_size=2, elitism_length=2) :

        self.population = np.array([ Chromosome(nn.n_features) for i in range(pop_size) ])
        self.nn = nn

        self.cross_rate = cross_rate
        self.pool_size = pool_size
        self.NUMBER_OF_GENS = n_gens
        self.elitism_length = elitism_length

        self.history = [[]]
        self.best = []
        self.present_features = []

        self.gen_step = gen_step

    """Calculates and returns the fitness for a specified chromosome.
    This is the first step after
    """
    def calc_fitness(self, chromosome, t=0.115, m=0.02) :
        chromosome_str = " ".join([str(x) for x in chromosome.vector])
        print("Training: [" + chromosome_str + "] ...")

        self.nn.train(chromosome.vector)
        print("Finished training...")

        accuracy = self.nn.percent_accuracy(chromosome.vector)[0]

        error_rate = 1.0 - accuracy
        fitness = np.sum(chromosome.vector) + (np.exp((error_rate - t) / m) - t) / (np.e - 1)

        print("Error rate (1-accuracy): " + str(error_rate))
        print("Fitness:" + str(fitness) + "\n\n")

        chromosome.accuracy = accuracy
        chromosome.fitness = fitness
        self.history[len(self.history) - 1].append(chromosome)

        return fitness

    """Recursive method that evolves the population to the next generation using:
        1. Elitism
        2. Selection (tournament)
        3. Crossover (mating)
        4. Mutation

    Usage: final = self.evolve(population)
        Note: Do not pass in a generation-leave it as the default param=0
    """
    def evolve(self, population=[], generation=0, verbose=False) :
        print("GENERATION " + str(generation) + " ...\n")

        _population = []
        chromosome_count = []
        added = False
        self.present_features.append([0] * self.nn.n_features)

        # Mark all used features
        for i in range(self.nn.n_features):
            for chromosome in self.population:
                if chromosome.vector[i] == 1:
                    self.present_features[len(self.present_features) - 1][i] = 1
                    break

        # Reinitialize base population
        for chromosome in self.population:
            for i in range(len(_population)):
                if(np.array_equal(chromosome.vector, _population[i].vector)):
                    chromosome_count[i] += 1
                    added = True
                    break

            if added is False:
                _population.append(chromosome)
                chromosome_count.append(1)

            added = False

        if verbose:
            print("Present Features: \n")
            for gen in self.present_features:
                print(gen)
            for chromosome, count in zip(_population, chromosome_count):
                print(str(chromosome.vector) + ": " + str(count))

        fitness_pop = population

        # Base Case
        if generation is 0:
            fitness_pop = [ (self.calc_fitness(chromosome), chromosome) for chromosome in self.population ]
        elif generation > self.NUMBER_OF_GENS or generation % self.gen_step == 0:
            return generation + 1, fitness_pop

        self.history.append([])
        pop_size = len(self.population)

        # Elitism
        sorted_pop = sorted(fitness_pop, key=lambda x: x[0])
        children = [elite for elite_fit, elite in sorted_pop[:self.elitism_length]]
        self.best.append(sorted_pop[0])

        if verbose:
            print(self.best + "\n")

        # Crossover & Mutation
        while len(children) < pop_size:
            male_fit, male = self._tournament(fitness_pop, self.pool_size)
            female_fit, female = self._tournament(fitness_pop, self.pool_size)

            print(str(male.vector) + ": " + str(male_fit))
            print(str(female.vector) + ": " + str(female_fit))

            _children = []
            if random.random() < self.cross_rate:
                _children = self.cross(generation, male, female)
            else:
                male.mutate()
                _children = [male]

            for child in _children:
                children.append(child)

        self.population = children[0:pop_size]

        if verbose:
            print("\nChildren:")

        # Assign fitness values to each solution
        fitness_pop = [ (self.calc_fitness(chromosome), chromosome) for chromosome in self.population ]

        return self.evolve(population=fitness_pop, generation=generation + 1)

    """Selects a solution based on its fitness value, using a pool size of k.
    """
    def _tournament(self, population, k) :
        sample = random.sample(population, k)
        return sorted(sample, key=lambda x: x[0])[0]

    """Crosses between two selected chromosomes, exchanging genetic information.
    IF (both chromosomes are the same) :
        Undergo adaptive chromosome replacement, replacing 1/2 solutions depending
        on the generation.
    """
    def cross(self, generation, male, female) :
        locuses = random.sample(range(0, male.length() - 1), 2)
        locus_left = min(locuses)
        locus_right = max(locuses)

        child_1 = deepcopy(male)
        child_2 = deepcopy(female)

        # Adaptive Chromosome Replacement (ACR)
        if child_1.fitness == child_2.fitness:
            if generation < int(self.NUMBER_OF_GENS / 3):
                child_1 = Chromosome(Trainer.n_features)
                child_2 = Chromosome(Trainer.n_features)
            elif generation < int(2 * self.NUMBER_OF_GENS / 3):
                child_2 = Chromosome(Trainer.n_features)
            else:
                child_2 = self.random_chromosome()

        child_1.replace_range(female, locus_left, locus_right)
        child_2.replace_range(male, locus_left, locus_right)

        return [child_1, child_2]

    """Chooses a random chromosome from the population.
    """
    def random_chromosome(self) :
        return r.choice(self.population)

    """Displays a time-series plot that shows how the mean accuracy of solutions
    changes from generation to generation.
    """
    def graph(self) :
        sns.set_style("darkgrid")

        average_accuracies = [sum([individual.accuracy for individual in generation]) / len(generation)
                                                       for generation in self.history]
        iters = range(len(average_accuracies))

        fig, ax = plt.subplots()
        ax.set_xlabel("Generations")
        ax.set_ylabel("Mean Accuracy")
        ax.set_xlim([0, 50])
        ax.plot(iters, average_accuracies)
        plt.show()
