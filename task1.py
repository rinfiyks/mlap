# Predict stock price for current day given stock price and stock volume for
# past 10 days

import sys
import csv
import numpy
import scipy.optimize


def calculate_loss(theta, volumes, prices):
    total_squared_loss = 0

    for row in xrange(10, len(volumes) - 1):
        recent_data = []  # last 10 days worth of data
        # first 10 elements are volumes, last 10 are prices,
        # in chronological order

        for i in range(row - 10, row):
            recent_data.append(volumes[i])
        for i in range(row - 10, row):
            recent_data.append(prices[i])

        total_squared_loss += (numpy.dot(theta, recent_data) - prices[row])**2

    print(total_squared_loss)
    return total_squared_loss


def linear(InputFileName):
    file_name = InputFileName

    volumes = []
    prices = []

    with open(file_name, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            volumes.append(float(row[0]))
            prices.append(float(row[1]))

    initial_theta = [0] * 20

    print("Starting BFGS...")
    theta = scipy.optimize.fmin(calculate_loss, x0=initial_theta,
                                args=(volumes, prices))
    print("BFGS finished.")
    print(theta)
    return 0

linear(sys.argv[1])
