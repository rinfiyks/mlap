# Predict stock price for current day given stock price and stock volume for
# past 10 days

import sys
import csv
import random
import numpy
import scipy.optimize

previous_days_count = 1  # the number of previous days given


def linear(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data(raw_data)

    initial_theta = [0] * len(row_data[0][1])

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    MSE1 = train_data(data_set_0, data_set_1, initial_theta)
    MSE2 = train_data(data_set_1, data_set_0, initial_theta)
    average_MSE = ((MSE1 + MSE2) / 2)

    print("Average MSE: %f" % average_MSE)
    return average_MSE


def train_data(training_data, testing_data, theta):
    theta = scipy.optimize.fmin(calculate_mean_squared_error, x0=theta, args=tuple([training_data]), disp=False)
    MSE = calculate_mean_squared_error(theta, testing_data)
    print(theta)
    print("MSE: %f" % MSE)
    return MSE


def calculate_mean_squared_error(theta, row_data):
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[1]) - row[0])**2

    total_squared_loss /= len(row_data)
    return total_squared_loss


def collate_row_data(raw_data):
    row_data = []

    for i in xrange(previous_days_count, len(raw_data)):
        row_data.append([])
        row_data[-1].append(float(raw_data[i][1]))  # append price

        features = []

        features += [float(raw_data[i - 1][1])]  # yesterday's price

        row_data[-1].append(features)

    return row_data


linear(sys.argv[1])
