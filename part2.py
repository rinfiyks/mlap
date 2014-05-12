from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy
import csv


def bnbayesfit(StructureFileName, DataFileName):

    structure = read_file(StructureFileName)
    datapoints = read_file(DataFileName)
    number_of_nodes = len(structure[0])

    parameters = []

    # for each node
    for node in range(number_of_nodes):
        # get list of parents, e.g. [1, 4]
        parents = get_parents(node, structure)

        # generate list [0, 0], the values of the parents for this iteration
        parent_values = [0] * len(parents)

        for x in range(2**len(parents)):
            node_values = [0.5, 0.5]  # the a priori alpha and beta

            for row in datapoints:
                bad_row = False
                for i in range(len(parents)):
                    if row[parents[i]] != parent_values[i]:
                        bad_row = True
                if not bad_row:
                    node_values[row[node]] += 1

            # add result to parameters
            parameters.append([node, beta(node_values[0], node_values[1]).mean(), parents, list(parent_values)])

            # lexicographically increment list of parent values
            parent_values = lexicographically_increment_list(parent_values)

    for param in parameters:
        print param

    return 0


def get_parents(node, structure):
    parents_structure = numpy.transpose(structure)
    parents = []
    for i in range(len(parents_structure[node])):
        if parents_structure[node][i] == 1:
            parents.append(i)
    return parents


def lexicographically_increment_list(l):
    decreasing = False
    for i in range(len(l)):
        if not decreasing:
            if l[i] == 0:
                l[i] = 1
                return l
            else:
                decreasing = True
                l[i] = 0
        elif l[i] == 0:
                l[i] = 1
                return l
        else:
            l[i] = 0

    return l


def read_file(file_name):
    data = []
    raw_data = list(csv.reader(open(file_name)))
    for row in raw_data:
        data.append([int(i) for i in row])
    return data


bnbayesfit("13/bnstruct.csv", "13/bndata.csv")


def testingstuff():
    rv = beta(5, 10)

    x = numpy.linspace(0, numpy.minimum(rv.dist.b, 3))

    h = plt.plot(x, rv.pdf(x))
    print rv.mean()

    plt.show(h)
