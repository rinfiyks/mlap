# Predict stock price for current day given stock price and stock volume for
# past 10 days

import sys
import csv
import numpy
import scipy.optimize

previous_days_count = 10  # the number of previous days given


def linear(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))

    initial_theta = [0] * 2 * previous_days_count

    row_data = collate_row_data(raw_data)

    training_row_data = row_data[0::2]
    testing_row_data = row_data[1::2]

    theta = scipy.optimize.fmin(calculate_mean_squared_error, x0=initial_theta,
                                args=tuple([training_row_data]))
    print("Theta:")
    print(theta)
    print("Testing data MSE:")
    print(calculate_mean_squared_error(theta, testing_row_data))
    return 0


def calculate_mean_squared_error(theta, row_data):
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[1]) - row[0])**2

    total_squared_loss /= len(row_data)

    print(total_squared_loss)
    return total_squared_loss


def collate_row_data(raw_data):
    row_data = []

    for i in xrange(10, len(raw_data)):
        row_data.append([])
        row_data[-1].append(float(raw_data[i][1]))  # append price

        previous_data = []

        for j in range(0, 10):
            previous_data.append(float(raw_data[i - j - 1][0]))
            previous_data.append(float(raw_data[i - j - 1][1]))
        row_data[-1].append(previous_data)

    return row_data


linear(sys.argv[1])
