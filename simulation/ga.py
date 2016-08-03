
import tensorflow as tf

import numpy as np
from numpy import random as r
from numpy.random import randint
from copy import deepcopy

import random as random
import matplotlib.pyplot as plt

import math
from enum import Enum

from nn import Trainer

class Chromosome(object):

    def __init__(self, n_features) :
        self.n_features = n_features

        self.vector = randint(2, size=n_features)

        self.accuracy = 0.0
        self.fitness = 0.0

    def mutate(self, iters=1) :
        for i in range(iters):
            mutation_index = randint(0, self.length() - 1)
            current = self.vector[mutation_index]
            self.vector[mutation_index] = 0 if current == 1 else 1

    def length(self) :
        return len(self.vector)

    def replace_range(self, other, start, end) :
        np.put(self.vector, range(start, end), other.vector[start:end])

class GA(object):

    """GA wrapper object handling NN feature selection

    Attributes:
        population: list of n (population size) randomly generated chromosomes
        nn: instantiated Trainer with training and testing data loaded
        cross_rate: probability that two parents will perform crossover to
                    produce a child
    """

    def __init__(self, nn, n_gens=8, pop_size=20, cross_rate=0.9, pool_size=3, elitism_length=2) :

        self.population = np.array([ Chromosome(nn.n_features) for i in range(pop_size) ])
        self.nn = nn

        self.cross_rate = cross_rate
        self.pool_size = pool_size
        self.NUMBER_OF_GENS = n_gens
        self.std_threshold = 0.02
        self.elitism_length = elitism_length

        self.history = []
        self.best = []
        self.best_counter = 0
        self.best_threshold = 5

        r.seed(49)

        self.unique_chromosomes = []

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        random.shuffle(l)
        step = len(l) / n
        for i in range(0, len(l), step):
            yield l[i:i+n]

    def calc_fitness(self, chromosome, t=0.115, m=0.02) :
        chromosome_str = " ".join([str(x) for x in chromosome.vector])
        print("Training: [" + chromosome_str + "] ...")

        # for i in range(len(self.unique_chromosomes)):
        #     if np.array_equal(chromosome.vector, self.unique_chromosomes[i][2].vector):
        #         print("Already exists...")
        #
        #         print("Error rate (1-accuracy): " + str(1 - self.unique_chromosomes[i][1]))
        #         print("Fitness:" + str(self.unique_chromosomes[i][0]) + "\n\n")
        #
        #         chromosome.accuracy = self.unique_chromosomes[i][1]
        #         chromosome.fitness = self.unique_chromosomes[i][0]
        #         return self.unique_chromosomes[i][0] # return fitness

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

        self.unique_chromosomes.append((fitness, accuracy, chromosome))

        return fitness

    """Recursive method that evolves the population to the next generation using:
        1. Selection (tournament)
        2. Crossover (mating)
        3. Mutation

    Usage: final = self.evolve(population)
        Note: Do not pass in a generation; leave it as the default param (0)
    """
    def evolve(self, population=[], generation=0) :
        print("GENERATION " + str(generation) + " ...\n")
        _population = []
        chromosome_count = []
        added = False

        for chromosome in self.population:
            for i in range(len(_population)):
                if(np.array_equal(chromosome.vector, _population[i])):
                    chromosome_count[i] += 1
                    added = True
                    break

            if added is False:
                _population.append(chromosome.vector)
                chromosome_count.append(1)

            added = False

        for chromosome, count in zip(_population, chromosome_count):
            print(str(chromosome) + ": " + str(count))

        fitness_pop = population
        if generation > self.NUMBER_OF_GENS:
            return

        self.history.append([])

        if generation is 0:
            fitness_pop = [ (self.calc_fitness(chromosome), chromosome) for chromosome in self.population ]

        pop_size = len(self.population)

        # Elitism

        sorted_pop = sorted(fitness_pop)
        children = [elite for elite_fit, elite in sorted_pop[:self.elitism_length]]

        # Judgement Day
        # Check if the gene pool has stagnated
        if len(self.best) > 0:
            if self.best[len(self.best) - 1][0] == sorted_pop[0][0]:
                self.best_counter += 1
                print("Same best in a row: " + str(self.best_counter))

        # Crossover & Mutation

        while len(children) < pop_size:
            male_fit, male = self._tournament(fitness_pop, self.pool_size)
            female_fit, female = self._tournament(fitness_pop, self.pool_size)

            _children = []
            if r.random() < self.cross_rate:
                print("\nCross: True")
                _children = self.cross(male, female)
            else:
                print("\nCross: False")
                male.mutate()
                _children = [male]

            for child in _children:
                children.append(child)

        self.population = children[0:pop_size]
        print("\nChildren:")

        # Social Disaster
        # _population = []
        # chromosome_count = []
        # added = False
        #
        # for chromosome in self.population:
        #     for i in range(len(_population)):
        #         if(np.array_equal(chromosome.vector, _population[i])):
        #             chromosome_count[i] += 1
        #             added = True
        #             break
        #
        #     if added is False:
        #         _population.append(chromosome.vector)
        #         chromosome_count.append(1)
        #
        #     added = False
        #
        # for chromosome, count in zip(_population, chromosome_count):
        #     print(str(chromosome) + ": " + str(count))
        #
        # for i in range(len(chromosome_count)):
        #     if chromosome_count[i] >= 3:
        #         if generation % 4 == 0:
        #             self.social_disaster(_population[i])
        #         else:
        #             self.social_disaster(_population[i], trim=True)

        self.best.append(sorted_pop[0])
        print(self.best)
        print("\n")

        fitness_pop = [ (self.calc_fitness(chromosome), chromosome) for chromosome in self.population ]

        # Adaptive Crossover Rate

        accuracies = [chromosome.accuracy for chromosome in self.history[len(self.history) - 1]]
        print(np.std(accuracies))
        if np.std(accuracies) < self.std_threshold and self.cross_rate > 0.6:
            self.cross_rate -= 0.1
            print("Cross rate: " + str(self.cross_rate))
        elif self.cross_rate < 0.9:
            self.cross_rate += 0.1

        return self.evolve(population=fitness_pop, generation=generation + 1)

    def _tournament(self, population, k) :
        sample = random.sample(population, k)
        return sorted(sample)[0]

    def _roulette(self, population) :
        sorted_inv_pop = self._norm_inv(population)
        max = sum(chromosome[0] for chromosome in sorted_inv_pop)
        print(max)
        rand_fit = random.uniform(0, max)
        current = 0
        for chromosome in sorted_inv_pop:
            current += chromosome[0]
            if current >= rand_fit:
                return chromosome

    def _norm_inv(self, population) :
        max_fit = max(population)[0]
        for chromosome in population:
            chromosome = (1 / chromosome[0], chromosome[1])
        return sorted(population)

    def cross(self, male, female) :
        locus_range = randint(5, 8)
        locus_left = randint(0, (male.length() - 1) - locus_range)
        locus_right = locus_left + locus_range

        # locuses = random.sample(range(0, male.length() - 1), 2)
        # locuses = [randint(0, male.length() - 1), randint(0, male.length() - 1)]
        # locus_left = min(locuses)
        # locus_right = max(locuses)

        print(locus_left, locus_right)

        child_1 = deepcopy(male)
        child_2 = deepcopy(female)

        # Random generation
        if np.array_equal(child_1.vector, child_2.vector):
            print("Randomly generating a child --------\n")
            child_2 = Chromosome(Trainer.n_features)
            return [child_1, child_2]

        child_1.replace_range(female, locus_left, locus_right)
        child_2.replace_range(male, locus_left, locus_right)

        print(child_1.vector, child_2.vector)

        return [child_1, child_2]

    def social_disaster(self, target, trim=False) :
        retained = False
        for i, chromosome in enumerate(self.population):
            if np.array_equal(chromosome.vector, target):
                if trim is True:
                    self.population[i].mutate(4)
                    break
                if retained is False:
                    retained = True
                    continue
                else:
                    self.population[i].mutate(4)

    def graph(self) :
        average_accuracies = [sum([individual.accuracy for individual in generation]) / len(generation) for generation in self.history]
        iters = range(len(average_accuracies))

        plt.plot(iters, average_accuracies)
        plt.show()

        plt.plot(iters, self.best)
        plt.show()
