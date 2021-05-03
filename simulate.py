# Dependencies
import numpy as np
import pandas as pd
import random
import operator

# Graph
import networkx as nx # To be replaced by plotly

import os

# Seed for random to ensure replicable results
random.seed(10) # MAKE SURE TO UNCOMMENT THIS LINE BEFORE RUNNING FINAL RESULTS FOR CONSISTENCY

import functools # Timer Function
import time as time


class Gene:
    """City class with attributes x and y coordinates"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
    
    def calcDistance(self, newGene):
        """
        Calcuate distance using pythagorean theorem agfains another city
        
        Parameters:
            newGene (object) city object 
            
        Returns:
            cartesianDistance (int) distance between current city and another
        """
        xAxis = abs(self.x - newGene.x)
        yAxis = abs(self.y - newGene.y)
        cartesianDistance = np.sqrt(pow(xAxis, 2) + pow(yAxis,2))
        return cartesianDistance

    def __repr__(self):
        """Cleaner way to represent city"""
        return f"({self.x}, {self.y})"


def gene_list(number_of_cities):
    """Creat list of random cities
    
    Parameters:
        number_of_cities (int) number of cities to make
        
    Returns:
        city_list (list) list of cities
    """
    city_list = []
    for i in range(number_of_cities):
        city_list.append(Gene(x=int(random.random() * 200), y=int(random.random() * 200)))
    return city_list


class Fitness:
    """Fitness class to determine the fitness of a given route"""
    def __init__(self, individual):
        self.individual = individual
        self.distance = 0
        self.fitness = 0.0
    
    def calcIndividualDistance(self):
        """
        Calculate route distance
        
        Returns:
            self.distance (int) route distance
        """
        if self.distance == 0:
            partialDistance = 0
            for i in range(0, len(self.individual)):
                startIndividual = self.individual[i]
                finishIndividual = None
                if i+1 < len(self.individual):    # Taking into account starting and finishing at the same city
                    finishIndividual = self.individual[i+1]
                else:
                    finishIndividual = self.individual[0]
                partialDistance = partialDistance + startIndividual.calcDistance(finishIndividual)
                self.distance = partialDistance
        return self.distance


    def calcIndivivualFitness(self):
        """
        Calculate fitness - larger fitness means a yeild of better results
        
        Returns:
            self.fitness (float) route fitness
        """
        if self.fitness == 0:
            self.fitness = 1 / float(self.calcIndividualDistance())
        return self.fitness


	
#funcion to create a random route of the length of Gene's 
def createIndividual(geneList):
    """
    Create initial route from random sample of cities matching lenght of number of cities
    
    Parameters:
        geneList (list) list of cities
        
    Returns: 
        individual (list)
    """
    individual = random.sample(geneList, len(geneList))
    return individual


def buildPopulation(size, geneList):
    """
    Create initial population
    
    Parameters:
        size (int) population size
        geneList (list) list of cities
        
    Returns:
        populationSet (list) population - list of routes the size of the population
    """
    populationSet = [createIndividual(geneList) for i in range(size)]
    return populationSet


def rankIndividuals(population):
    """Rank routes withinb population by fitness to simulate survival of the fittest
    
    Paramters:
        population (list) list of routes
        
    Returns:
        (list) ordered list of route IDs and associated fitness
    """
    results= {}
    for i in range(len(population)):
        results[i] = Fitness(population[i]).calcIndivivualFitness() 
    return sorted(results.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    """
    Lines 6-8 setup roulette wheel by calculationg relative fitness weight for each individual
    
    Paramter5s:
        popRanked (list) list of ranked populations
        eliteSize (int) number or elites to be carried over
        
    Returns:
        selectionResults (list) list of routes to select in selection process. 
    """
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(eliteSize):  # returns list of route IDs to create mating pool in buildMatingPool func
        selectionResults.append(popRanked[i][0])
    for i in range(len(popRanked) - eliteSize):  # returns list of route IDs to create mating pool in buildMatingPool func
        pick = 100*random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def buildMatingPool(population, selectionResults):
    """Extract selected individuals for mating
    
    Parameters: 
        population (list) list of routes
        selectionResults (list) selected individuals
    """
    pool = []
    length = len(selectionResults)
    for i in range(length):
        index = selectionResults[i]
        pool.append(population[index])
    return pool

def breed(parent1, parent2):
    """Use ordered crossover to create child
    
    Paramters:
        parent1 (list)
        parent2 (list)
        
    Returns:
        child (list1)
    """
    childP1, childP2 = [], []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    """Create crossover of full mating pool
    
    Parameters:
        matingpool (list)
        eliteSize (int)
        
    Returns:
        children (list)
    """
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    """ Mutate paths within solution space to avoid local convergance
    
    Parameters: 
        individual (list)
        mutationRate (float) percentage of mutation
        
    Returns:
        individual (list)
    """
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    """Mutate new population
    
    Paramters:
        population (list)
        mutationRate (float)percentage or mutation
        
    Returns:
        mutatedPop (list)
        """
    mutatedPop = []
    
    for individual in range(len(population)):
        mutatedInd = mutate(population[individual], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    """Create next generation
    
    Parameters:
        currentGen (list)
        eliteSize (int)
        mutationRate (float)
    
    Returns:
        nextGeneration (list)
    """
    popRanked = rankIndividuals(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingPool = buildMatingPool(currentGen, selectionResults)
    children = breedPopulation(matingPool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    """ Find most opimal route
    
    Parameters:
        population (list)
        popSize (int)
        eliteSize (int)
        mutationRate (float)
        generation (int)
        
    Returns:
        bestRoute
    """
    time_per_generation = []
    t_1, num_1, t_orig_1 = 0, 0, time.perf_counter() # START TIMER
    progress = []
    pop = buildPopulation(popSize, population)
#     print("Initial distance: " + str(1 / rankIndividuals(pop)[0][1]))
    print(f"Initial distance: {1 / rankIndividuals(pop)[0][1]}")
    
    for i in range(generations):
        print(f"Begining Geenration: {i}")
        t, num, t_orig = 0, 0, time.perf_counter()
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankIndividuals(pop)[0][1])
        t = time.perf_counter() # END TIMER
        time_per_generation.append(t - t_orig)
    
#     print("Final distance: " + str(1 / rankIndividuals(pop)[0][1]))
    print(f"Final distance: {1 / rankIndividuals(pop)[0][1]}")
    bestRouteIndex = rankIndividuals(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    
    t_1 = time.perf_counter() # END TIMER
    execution_time = t_1 - t_orig_1 # RETUNR DIRERENCE - AKA EXECUTION TIME
    return bestRoute, progress, execution_time, time_per_generation


def routetodict(route, dataframe):
    """Placew a route inside a dictionary
    
    Parameters:
        route (list of cities)
    
    Returns:
        routes_dict (dict) dictrionary of routes stored as tuples
        dataframe (pandas dataframe) 
    """
    routes_dict = {}
    city_index = 0
    
    for index in route:
#         routes_dict[city_index] = (index.x, index.y)
        
        dataframe.loc[city_index] = [index.x, index.y]
        city_index += 1
    
    return dataframe


def create_networkx_graph(tour):
    """Create a netowkrx graph from dataframe
    
    Paramters: 
        tour (pandas dataframe) Dataframe containing ordered cities/tour
    
    Returns:
        tour_graph (object) Networkx Graph Object
    """
    tour_graph = nx.Graph()
    
    for index, row in tour.iterrows():
        tour_graph.add_node(index, pos=(row["city_from"], row["city_to"]))
        
    for index in range(len(tour_graph.nodes)):
        if index == len(tour_graph.nodes) - 1:
            tour_graph.add_edge(index, 0)
        else:
            tour_graph.add_edge(index, index+1)
            
    return tour_graph



def dataframe_to_csv(filepath, filename, route, progress, time_per_generation):
    """Write dataframes to .csv in a chosen directory following specified naming convention
    
    Parameters:
        filepath (string) relative file path
        filename (string) filename convention
        roue_df (dataframe) optimal route/tour dataframe
        progress_df (dataframe) GA distances per generation dataframe
    """

    # check if filepath exists and create directory if not
    if not os.path.exists(filepath):
        try:
            os.makedirs(filepath)
        except Exception as e:
            raise Exception("Unable to make director: %s \n %s", (filepath, e))
            
    # Convert to dataframes
    column_names = ["city_from", "city_to"]
    cities = pd.DataFrame(columns = column_names)

    route_df =  routetodict(route, cities) # Use this df for creating graph of final route
    progress_df = pd.DataFrame(progress, columns=["Distance"])
    progress_df["Time"] = pd.Series(time_per_generation).values

    # Write dataframes to csv using naming convention                    
    route_df.to_csv(f"{filepath}/{filename}_route.csv")
    progress_df.to_csv(f"{filepath}/{filename}_progress.csv")
                            

def simulate_tsp(configuration_dict, total_exectution_time):
    print("starting Simulation")
    route, progress, total_time, time_per_generation = geneticAlgorithm(population=gene_list(configuration_dict["gene_list"]), 
                                                                        popSize=configuration_dict["population_size"], 
                                                                        eliteSize=configuration_dict["elites"], 
                                                                        mutationRate=configuration_dict["mutation_rate"], 
                                                                        generations=configuration_dict["generations"])
    total_exectution_time = total_exectution_time.append({"population" : configuration_dict["gene_list"], "execution_time": round(total_time)}, ignore_index=True)
    dataframe_to_csv(configuration_dict["directory"], configuration_dict["filename_convention"], route, progress, time_per_generation)  # Create CSV files from output - UN-COMMENT THIS WHEN YOU WANT TO GEENRATE CSV FILES
    print(total_exectution_time)
    print(f"Simulation completed in: {round(total_time)} (Sec)")
    return total_exectution_time


# ////////////////////////////////////////////////// Run \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
total_exectution_time = pd.DataFrame(columns=["population", "execution_time"])



# 25 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 25, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")
# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 30 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 30, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")
# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 50 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 50, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# 150 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 150, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# 200 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 200, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 250 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 250, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")
# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 300 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 300, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")
# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 400 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 400, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")
# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 500 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 500, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 1000 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 1000, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "cities"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")


# /////////////////////////// mutation \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# 100 cities, 100 population, 20 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.01, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 2.5% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.025, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 5% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.05, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")


# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 7.5% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.075, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 10% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.10, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)
# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 15% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.15, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)

# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

# /////////////////////////////// ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 100 cities, 100 population, 20 elites, 20% mutation, 500 generations
configuration_dict = {"gene_list": 100, "population_size": 100, "elites": 20, "mutation_rate": 0.20, "generations": 500, "directory": "mutation"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

total_exectution_time =  simulate_tsp(configuration_dict, total_exectution_time)

# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")


# /////////////////////////////// Overkill \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

overkill_total_exectution_time = pd.DataFrame(columns=["population", "execution_time"])

# 2500 cities, 980 population, 100 elites, 1% mutation, 500 generations
configuration_dict = {"gene_list": 2500, "population_size": 980, "elites": 100, "mutation_rate": 0.01, "generations": 500, "directory": "cities_overkill"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

overkill_total_exectution_time =  simulate_tsp(configuration_dict, overkill_total_exectution_time)

# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

overkill_total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")

overkill_total_exectution_time = pd.DataFrame(columns=["population", "execution_time"])

# 2500 cities, 980 population, 100 elites, 1% mutation, 3000 generations
configuration_dict = {"gene_list": 2500, "population_size": 980, "elites": 100, "mutation_rate": 0.01, "generations": 3000, "directory": "cities_overkill"}
configuration_dict["filename_convention"] = f"GA_TSP_{configuration_dict['gene_list']}city_{configuration_dict['population_size']}popsize_{configuration_dict['elites']}elite_{str(configuration_dict['mutation_rate']).replace('.', '')}mut_{configuration_dict['generations']}gen"
configuration_dict["filename_convention"]

overkill_total_exectution_time =  simulate_tsp(configuration_dict, overkill_total_exectution_time)

# /////////////////////////////// Save CSV \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

overkill_total_exectution_time.to_csv(f"{configuration_dict['directory']}/{configuration_dict['filename_convention']}_total_time.csv")