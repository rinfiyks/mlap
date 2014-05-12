from scipy.stats import beta
import itertools
import numpy
import csv


def bnbayesfit(StructureFileName, DataFileName):
    structure = read_file(StructureFileName)
    datapoints = read_file(DataFileName)
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
            node_values = [0.5, 0.5]  # the a priori alpha and beta

            # iterate over each line in the datapoints file
            for row in datapoints:
                good_row = True  # whether or not this datapoint has correct values for the parents
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


bnbayesfit("13/bnstruct.csv", "13/bndata.csv")
