# Predict stock price for current day given stock price and stock volume for
# past 10 days

import sys
import csv
import random
import numpy
import scipy.optimize

previous_days_count = 10  # the number of previous days given
number_of_classes = 5  # the number of classes used by logistic function


def linear(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_linear(raw_data)

    initial_theta = [0] * len(row_data[0][-1])

    return regression(row_data, initial_theta, calculate_mean_squared_error)


def logistic(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_logistic(raw_data)
    initial_theta = [0] * (len(row_data[0][-1]) * number_of_classes)

    return regression(row_data, initial_theta, calculate_mean_classification_error)


def regression(row_data, initial_theta, error_function):
    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    error1 = train_data(data_set_0, data_set_1, initial_theta, error_function)
    error2 = train_data(data_set_1, data_set_0, initial_theta, error_function)
    average_error = ((error1 + error2) / 2)

    print("Average error: %f" % average_error)
    return average_error


def train_data(training_data, testing_data, theta, error_function):
    theta = scipy.optimize.fmin(error_function, x0=theta, args=tuple([training_data]), disp=False)
    error = error_function(theta, testing_data)
    print(theta)
    print("error: %f" % error)
    return error


def calculate_mean_squared_error(theta, row_data):
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[-1]) - row[0])**2

    total_squared_loss /= float(len(row_data))
    return total_squared_loss


def calculate_mean_classification_error(theta, row_data):
    total_error = 0

    for row in row_data:
        # p(y = i|X) = e^(XT_i) / sum_j(e^(XT_j))
        # y = today's change
        # i = row[2] = class
        # X = row[-1] = features
        # T = theta

        # c + log(sum_i(e^(x_i - c)))
        features_size = len(row[-1])

        class_theta = theta[row[2] * features_size:(row[2] + 1) * features_size]  # Theta_c

        max_feature = max(row[-1])
        thingy = []
        for i in range(0, features_size):
            thingy.append(numpy.dot([j - max_feature for j in row[-1]], theta[i * features_size:(i + 1) * features_size]))

        total_error += numpy.dot(class_theta, row[-1])
        - max_feature - scipy.misc.logsumexp(thingy)
        # - scipy.misc.logsumexp(numpy.dot(theta, row[-1] * number_of_classes))

        # classification = classify_day(numpy.dot(theta, row[-1] * 5), row[1])

    print -total_error
    return -total_error


def classify_day(today, yesterday):
    stock_difference = (today - yesterday) / yesterday

    classification = 0
    if stock_difference < -0.1:
        classification = 4
    elif stock_difference < -0.05:
        classification = 2
    elif stock_difference <= 0.05:
        classification = 0
    elif stock_difference <= 0.1:
        classification = 1
    else:
        classification = 3

    return classification


def get_features(rows):
    features = []
    features += [float(rows[-1][1])]  # yesterday's price
    #  features += [1]  # constant
    return features


def collate_row_data_linear(raw_data):
    row_data = []

    for i in xrange(previous_days_count, len(raw_data)):
        row_data.append([])  # add a new row

        row_data[-1].append(float(raw_data[i][1]))  # add today's price to required data

        features = get_features(raw_data[i - previous_days_count:i])
        row_data[-1].append(features)  # append features to row

    # each row contains [today's price, [features]]
    return row_data


def collate_row_data_logistic(raw_data):
    row_data = []

    for i in xrange(previous_days_count, len(raw_data)):
        row_data.append([])  # add a new row

        row_data[-1].append(float(raw_data[i][1]))  # add today's price to required data
        row_data[-1].append(float(raw_data[i - 1][1]))  # add yesterday's price to required data

        stock_today = float(raw_data[i][1])
        stock_yesterday = float(raw_data[i - 1][1])
        classification = classify_day(stock_today, stock_yesterday)

        row_data[-1].append(classification)  # append class to required data

        features = get_features(raw_data[i - previous_days_count:i])
        row_data[-1].append(features)  # append features to row

    # each row contains [today's price, yesterday's price, class, [features]]
    return row_data


logistic(sys.argv[1])
