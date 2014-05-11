# Predict stock price for current day given stock price and stock volume for
# past 10 days

import sys
import csv
import random
import math
import numpy
import scipy.optimize

previous_days_count = 10  # the number of previous days given
number_of_classes = 5  # the number of classes used by logistic function


def linear(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_linear(raw_data)

    initial_theta = [0] * len(row_data[0][-1])

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    error_0 = train_data(data_set_0, data_set_1, initial_theta,
                         calculate_squared_error, calculate_squared_error_gradient)[0]
    error_1 = train_data(data_set_1, data_set_0, initial_theta,
                         calculate_squared_error, calculate_squared_error_gradient)[0]

    MSE_0 = error_0 / float(len(data_set_0))
    MSE_1 = error_1 / float(len(data_set_1))

    average_error = (MSE_0 + MSE_1) / 2
    print("Average MSE: %f" % average_error)
    return average_error


def logistic(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_logistic(raw_data)
    initial_theta = [0] * (len(row_data[0][-1]) * number_of_classes)

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    theta0 = train_data(data_set_0, data_set_1, initial_theta,
                        calculate_classification_error, calculate_classification_error_gradient)[1]
    theta1 = train_data(data_set_1, data_set_0, initial_theta,
                        calculate_classification_error, calculate_classification_error_gradient)[1]

    accuracy_0 = calculate_predictor_accuracy(theta0, data_set_1)
    accuracy_1 = calculate_predictor_accuracy(theta1, data_set_0)

    average_accuracy = (accuracy_0 + accuracy_1) / 2.0
    print("accuracy_0: %f" % accuracy_0)
    print("accuracy_1: %f" % accuracy_1)
    print("average_accuracy: %f" % average_accuracy)
    return 0


def reglinear(InputFileName):
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_linear(raw_data)

    initial_theta = [0] * len(row_data[0][-1])

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    lambda_best = -1
    error_best = float("Inf")
    iterations = 10

    for i in range(iterations):
        lambda_param = 0.1 - (0.01 * i)

        error_0 = train_data_ridge(data_set_0, data_set_1, initial_theta, lambda_param,
                                   calculate_squared_error_ridge, calculate_squared_error_ridge_gradient)[0]
        error_1 = train_data_ridge(data_set_1, data_set_0, initial_theta, lambda_param,
                                   calculate_squared_error_ridge, calculate_squared_error_ridge_gradient)[0]

        MSE_0 = error_0 / float(len(data_set_0))
        MSE_1 = error_1 / float(len(data_set_1))

        average_error = (MSE_0 + MSE_1) / 2
        if average_error < error_best:
            error_best = average_error
            lambda_best = lambda_param

        print("Lambda: %.2f" % lambda_param)
        print("Error: %f" % average_error)

    print("Best lambda: %.2f" % lambda_best)

    return error_best


def calculate_predictor_accuracy(theta, row_data):
    correct = 0
    incorrect = 0

    for row in row_data:
        predicted_class = predict_class(row, theta)
        if predicted_class == row[2]:
            correct += 1
        else:
            incorrect += 1

    return correct / float(correct + incorrect)


def train_data(training_data, testing_data, theta, error_function, gradient_function):
    # theta = scipy.optimize.fmin(error_function, x0=theta, args=tuple([training_data]), disp=False)
    theta = scipy.optimize.fmin_bfgs(error_function, x0=theta,
                                     args=tuple([training_data]), disp=False, fprime=gradient_function)

    error = error_function(theta, testing_data)
    print(theta)
    return (error, theta)


def train_data_ridge(training_data, testing_data, theta, lambda_param, error_function, gradient_function):
    # theta = scipy.optimize.fmin(error_function, x0=theta, args=tuple([training_data, lambda_param]), disp=False)
    theta = scipy.optimize.fmin_bfgs(error_function, x0=theta,
                                     args=tuple([training_data, lambda_param]), disp=False, fprime=gradient_function)

    error = error_function(theta, testing_data, lambda_param)
    print(theta)
    return (error, theta)


def calculate_squared_error(theta, row_data):
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[-1]) - row[0])**2

    return total_squared_loss


def calculate_squared_error_gradient(theta, row_data):
    gradient = [0] * len(row_data[0][-1])

    for row in row_data:
        gradient += numpy.dot(row[-1], numpy.dot(theta, row[-1]) - row[0]) * 2

    return gradient


def calculate_squared_error_ridge(theta, row_data, lambda_param):
    loss_term = 0

    for row in row_data:
        loss_term += (numpy.dot(theta, row[-1]) - row[0])**2

    theta_term = sum(theta**2)

    print
    print("lambda: %f" % lambda_param)
    print("lambda*loss")
    print(lambda_param * loss_term)
    print("1-lambda *thetas**2")
    print((1 - lambda_param) * theta_term)
    return lambda_param * loss_term + (1 - lambda_param) * theta_term


def calculate_squared_error_ridge_gradient(theta, row_data, lambda_param):
    loss_term = [0] * len(row_data[0][-1])

    for row in row_data:
        loss_term += numpy.dot(row[-1], row[0] - (numpy.dot(theta, row[-1])))

    gradient = 2 * ((1 - lambda_param) * theta - lambda_param * loss_term)
    return gradient


def calculate_classification_error(theta, row_data):
    total_sum = 0

    for row in row_data:
        features = row[-1]
        features_size = len(features)
        current_class = row[2]

        max_feature = max(features)
        logsumexp_term = 0
        for i in range(0, number_of_classes):
            logsumexp_term += math.exp(numpy.dot(features, get_theta_for_class(theta, i, features_size)) - max_feature)
        logsumexp_term = math.log(logsumexp_term)

        X_Theta_c_term = numpy.dot(features, get_theta_for_class(theta, current_class, features_size))

        total_sum += max_feature + logsumexp_term - X_Theta_c_term

    return total_sum


def calculate_classification_error_gradient(theta, row_data):
    number_of_features = len(row_data[0][-1])
    gradient_dimension = number_of_features * number_of_classes
    gradient = [0] * gradient_dimension

    for row in row_data:
        features = row[-1]
        features_size = len(features)

        max_feature = max(features)
        logsumexp_term = [0] * gradient_dimension

        for i in range(0, number_of_classes):
            logsumexp_term += numpy.dot(features, math.exp(numpy.dot(features, get_theta_for_class(theta, i, features_size)) - max_feature))

        logsumexp_term = numpy.log(logsumexp_term)

        gradient += (max_feature + logsumexp_term) - features

    print(gradient)
    return gradient


def predict_class(row, theta):
    max = 0
    max_class = -1
    probabilities = []
    for c in range(0, number_of_classes):
        numerator = math.exp(numpy.dot(row[-1], get_theta_for_class(theta, c, len(row[-1]))))
        denominator = 0
        for i in range(0, number_of_classes):
            denominator += math.exp(numpy.dot(row[-1], get_theta_for_class(theta, i, len(row[-1]))))
        result = numerator / denominator
        probabilities.append(result)
        if result > max:
            max = result
            max_class = c
    return max_class


def get_theta_for_class(theta, c, features_size):
    start_index = features_size * c
    end_index = features_size * (c + 1)
    return theta[start_index:end_index]


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

    # for row in rows:
    #     features += [float(row[1])]
    features += [float(rows[-1][1])]
    features += [1]  # constant

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


print(reglinear(sys.argv[1]))
