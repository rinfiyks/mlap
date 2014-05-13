from scipy.stats import beta
import random
import itertools
import numpy
import csv


def bnbayesfit(StructureFileName, DataFileName):
    structure = read_file(StructureFileName)
    datapoints = read_file(DataFileName)
    return fit(structure, datapoints)


def fit(structure, datapoints):
    number_of_nodes = len(structure[0])

    parameters = []

    # for each node:
    for node in range(number_of_nodes):

        # get list of parents
        parents = get_parents(node, structure)

        # generate values parents could be, e.g. if there are 2 parents, this
        # would be [(0, 0), (0, 1), (1, 0), (1, 1)]
        parent_values_permutations = list(itertools.product((0, 1), repeat=len(parents)))

        # iterate over the values the parents can be
        for parent_values in parent_values_permutations:
            node_values = [1, 1]  # the uniform prior for alpha and beta

            # iterate over each line in the datapoints file
            for row in datapoints:
                good_row = True  # whether or not this datapoint has correct values for the parents

                # for each parent, check it has the correct value
                for i in range(len(parents)):
                    if row[parents[i]] != parent_values[i]:
                        good_row = False

                if good_row:
                    # if the values for the parents are correct, increment the alpha or beta value by 1
                    node_values[1 - row[node]] += 1

            # add result to parameters
            parameters.append([node, beta(node_values[0], node_values[1]).mean(), parents, list(parent_values)])

    for param in parameters:
        print param

    return parameters


def get_parents(node, structure):
    parents_structure = numpy.transpose(structure)
    parents = []
    for i in range(len(parents_structure[node])):
        if parents_structure[node][i] == 1:
            parents.append(i)
    return parents


def read_file(file_name):
    data = []
    raw_data = list(csv.reader(open(file_name)))
    for row in raw_data:
        data.append([int(i) for i in row])
    return data


def bnsample(fittedbn, nsamples):
    parents = []

    for param in fittedbn:
        if len(parents) == param[0]:
            parents.append(param[2])

    # the order in which the values for variables are generated
    # a node's value must not be generated until its parents have values
    generation_order = find_generation_order(parents)

    samples = []

    for n in range(nsamples):
        generated_values = [0] * len(parents)

        for gen in generation_order:  # generate in the correct order
            for param in fittedbn:  # attempt to find param with correct value for node and parents

                # make sure the param is for the correct node and parent values
                if not matching_param(param, generated_values, gen):
                    continue

                generated_value = random.uniform(0, 1)
                if generated_value < param[1]:
                    generated_value = 1
                else:
                    generated_value = 0

                generated_values[gen] = generated_value

        samples.append(generated_values)

    # fit(read_file("13/bnstruct.csv"), samples)

    return samples


def find_generation_order(parents):
    # calculate a correct generation order, where parents are generated first
    generation_order = []
    for gen in range(len(parents)):
        for i in range(len(parents)):
            if are_parents_generated(parents[i], generation_order) and i not in generation_order:
                generation_order.append(i)
                break
    return generation_order


def are_parents_generated(parents, generated):
    generated = set(generated)
    difference = [i for i in parents if i not in generated]
    return len(difference) == 0


def matching_param(param, generated_values, gen):
    if param[0] != gen:
        return False

    for p in range(len(param[2])):  # iterate over length of the node's parents
        if param[3][p] != generated_values[param[2][p]]:  # if the parent's value doesn't match
            return False
    return True


f = bnbayesfit("13/bnstruct.csv", "13/bndata.csv")
# bnsample(f, 10)
