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

    print("Starting fmin...")
    theta = scipy.optimize.fmin(calculate_mean_squared_error, x0=initial_theta,
                                args=tuple([row_data]))
    print("fmin finished.")
    print(theta)
    return 0


def calculate_mean_squared_error(theta, row_data):
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[2] + row[3]) - row[1])**2

    total_squared_loss /= len(row_data)

    print(total_squared_loss)
    return total_squared_loss


def collate_row_data(raw_data):
    row_data = []

    for i in xrange(10, len(raw_data)):
        row_data.append([])
        row_data[-1].append(float(raw_data[i][0]))  # append volume
        row_data[-1].append(float(raw_data[i][1]))  # append price

        volumes = []
        prices = []

        for j in range(0, 10):
            volumes.append(float(raw_data[i - j - 1][0]))
            prices.append(float(raw_data[i - j - 1][1]))
        row_data[-1].append(volumes)
        row_data[-1].append(prices)

    return row_data


linear(sys.argv[1])
