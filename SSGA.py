
#####################################################
# discard
#####################################################
# -*- coding: utf-8 -*-
"""
@author Linyuan Zhang


import os
import random
import sys
import numpy as np
import newBaseline as base
#
#  A Simple Steady-State, Real-Coded Genetic Algorithm
#  with uniform crossover, popsize = 25, chromosome_length = 95

class anIndividual:
    def __init__(self, specified_chromosome_length, group_size):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length
        self.popsize = popsize
        self.group_size = group_size
        self.temp = np.zeros([self.group_size,self.chromosome_length])

    def randomly_generate(self):

        temp1 = []
        for i in range(self.group_size):
            for j in range(self.chromosome_length):
                temp1.append(random.randint(0,1))
            self.chromosome.append(temp1)

        #[100, 95]
        #self.temp = np.zeros([self.group_size,self.chromosome_length])
        for i in range(self.group_size):
            for j in range(self.chromosome_length):
                self.temp[i][j] = random.randint(0,1)
        for ii in range(self.group_size):
            self.chromosome.append(self.temp[ii])
        self.fitness = 0

    def calculate_fitness(self):
        self.fitness = base.calculate_acc(self.temp)

    def print_individual(self, i):
        #print("Chromosome "+str(i) +": " + str(self.chromosome) + " Fitness: " + str(self.fitness))
        print("Chromosome "+str(i) +" Fitness: " + str(self.fitness))


class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate, group_size, k):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.population = []
        self.group_size = group_size
        self.k = k

    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length, self.group_size)
            individual.randomly_generate()
            individual.calculate_fitness()

            self.population.append(individual)

    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness):
                worst_fitness = self.population[i].fitness
                worst_individual = i
        return worst_individual

    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness

    def tournament_selection(self, k):
        parents = random.sample(self.population, k)
        best_P = parents[0]
        for item in parents:
            if item.fitness > best_P.fitness:
                best_P = item
        return best_P

    def evolutionary_cycle(self, k):

        mom = random.randint(0,self.population_size-1)
        dad = random.randint(0,self.population_size-1)
        kick = self.get_worst_fit_individual()
        p = random.randint(0, self.chromosome_length - 1)
        for i in range(self.group_size):
            for j in range(self.chromosome_length):
                if (p == 0):
                    self.population[kick].chromosome[i][j] = self.population[mom].chromosome[i][j]
                else:
                    self.population[kick].chromosome[i][j] = self.population[dad].chromosome[i][j]
                if np.random.uniform(0,1) < self.mutation_amt:
                    self.population[kick].chromosome[i][j] = not self.population[kick].chromosome[i][j]

        mom = self.tournament_selection(k)
        dad = self.tournament_selection(k)
        kick = self.get_worst_fit_individual()
        p = random.randint(0, self.chromosome_length - 1)
        for i in range(self.group_size):
            for j in range(self.chromosome_length):
                if (p == 0):
                    self.population[kick].chromosome[i][j] = mom.chromosome[i][j]
                else:
                    self.population[kick].chromosome[i][j] = dad.chromosome[i][j]
                if np.random.uniform(0,1) < self.mutation_amt:
                    self.population[kick].chromosome[i][j] = not self.population[kick].chromosome[i][j]

        self.population[kick].calculate_fitness()


    def uniform_crossover(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate()
        for i in range(self.chromosome_length):
            kid[i] = mom[i] if np.random.choice([True, False]) else dad[i]
            if np.random.uniform(0,1) < 0.05:
                kid[i] = not kid[i]
        return kid

    def print_population(self):
        for i in range(self.population_size):
            self.population[i].print_individual(i)

    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        #print("Best Indvidual: ",str(best_individual)," ", self.population[best_individual].chromosome, " Fitness: ", str(best_fitness))
        return self.population[best_individual].chromosome, best_fitness


ChromLength = 95
MaxEvaluations = 4000
plot = 0

popsize = 7
group_size = 100
mu_amt  = 0.01
k = 4
feature_mask = []
acc = []

ssga = aSimpleExploratoryAttacker(popsize,ChromLength,mu_amt,group_size,k)

ssga.generate_initial_population()
#ssga.print_population()

for i in range(MaxEvaluations-popsize+1):
    ssga.evolutionary_cycle(k)
    if (i % popsize == 0):
        print("At Iteration: " + str(i))
        ssga.print_population()
    if (ssga.get_best_fitness() >= 0.6):
        break

print("\nFinal Population\n")
#ssga.print_population()
mask, best_acc = ssga.print_best_max_fitness()
feature_mask.append(mask)
acc.append(best_acc)
print("\nmask: ", str(feature_mask), "\nacc: ", str(acc))
print("Function Evaluations: " + str(popsize+i))


for i in range(30):
    ssga = aSimpleExploratoryAttacker(popsize,ChromLength,mu_amt,group_size,k)
    ssga.generate_initial_population()

    for i in range(MaxEvaluations-popsize+1):
        ssga.evolutionary_cycle(k)
        if (i % popsize == 0):
            print("At Iteration: " + str(i))
            ssga.print_population()
        if (ssga.get_best_fitness() >= 0.9975):
            break

    print("\nFinal Population\n")
    ssga.print_population()
    ssga.print_best_max_fitness()

"""
from scipy import stats
import matplotlib.pyplot as plt

lis = []
lis2 = []
rbf = "0.84	0.84	0.76	0.8	0.84	0.88	0.76	0.84	0.8	0.8	0.8	0.8	0.84	0.76	0.8	0.64	0.84	0.8	0.72	0.8	0.84	0.8	0.72	0.8	0.88	0.76	0.76	0.8	0.8	0.8"
cha = rbf.strip().split("\t")
for i in cha:
    a = float(i)
    lis.append(a)
print("rbf: ", lis)

lsvm = "0.8	0.88	0.92	0.92	0.84	0.76	0.84	0.8	0.76	0.84	0.84	0.88	0.84	0.88	0.68	0.84	0.84	0.88	0.8	0.88	0.84	0.8	0.8	0.8	0.84	0.76	0.84	0.76	0.8	0.76"
cha2 = lsvm.strip().split("\t")
for i2 in cha2:
    a2 = float(i2)
    lis2.append(a2)
print("\nlsvm: ",lis2)

result = stats.ttest_ind(lis, lis2)
print("\nt-test result: ", result)

