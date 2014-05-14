import sys
import csv
import random
import math
import numpy
import scipy.optimize

previous_days_count = 10  # the number of previous days given
number_of_classes = 5  # the number of classes used by logistic function


def linear(InputFileName):
    """Perform linear regression on the data provided in InputFileName.
    """
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

    MSE_0 = error_0 / float(len(data_set_1))
    MSE_1 = error_1 / float(len(data_set_0))

    average_error = (MSE_0 + MSE_1) / 2
    return average_error


def reglinear(InputFileName):
    """Perform linear regression with ridge regularisation on the data provided in InputFileName.
    """
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_linear(raw_data)

    initial_theta = [0] * len(row_data[0][-1])

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    lambda_best = -1
    error_best = float("Inf")

    for i in range(0, 100):
        lambda_param = i / float(100)

        error_0 = train_data_ridge(data_set_0, data_set_1, initial_theta, lambda_param,
                                   calculate_squared_error_ridge, calculate_squared_error_ridge_gradient, calculate_squared_error)[0]
        error_1 = train_data_ridge(data_set_1, data_set_0, initial_theta, lambda_param,
                                   calculate_squared_error_ridge, calculate_squared_error_ridge_gradient, calculate_squared_error)[0]

        MSE_0 = error_0 / float(len(data_set_1))
        MSE_1 = error_1 / float(len(data_set_0))

        average_error = (MSE_0 + MSE_1) / 2
        if average_error < error_best:
            error_best = average_error
            lambda_best = lambda_param

    return error_best


def logistic(InputFileName):
    """Perform logistic regression on the data provided in InputFileName.
    """
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_logistic(raw_data)
    initial_theta = [0] * (len(row_data[0][-1]) * number_of_classes)

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    theta_0 = train_data(data_set_0, data_set_1, initial_theta,
                         calculate_classification_error, calculate_classification_error_gradient)[1]
    theta_1 = train_data(data_set_1, data_set_0, initial_theta,
                         calculate_classification_error, calculate_classification_error_gradient)[1]

    accuracy_0 = calculate_predictor_accuracy(theta_0, data_set_1)
    accuracy_1 = calculate_predictor_accuracy(theta_1, data_set_0)

    average_accuracy = (accuracy_0 + accuracy_1) / 2.0

    return average_accuracy


def reglogistic(InputFileName):
    """Perform logistic regression with ridge regularisation on the data provided in InputFileName.
    """
    raw_data = list(csv.reader(open(InputFileName, 'rU')))
    row_data = collate_row_data_logistic(raw_data)
    initial_theta = [0] * (len(row_data[0][-1]) * number_of_classes)

    random.shuffle(row_data)
    data_set_0 = row_data[0::2]
    data_set_1 = row_data[1::2]

    lambda_best = -1
    best_accuracy = 0

    for i in range(0, 100):
        lambda_param = i / float(100)

        theta_0 = train_data_ridge(data_set_0, data_set_1, initial_theta, lambda_param,
                                   calculate_classification_error_ridge, calculate_classification_error_ridge_gradient, calculate_classification_error)[1]
        theta_1 = train_data_ridge(data_set_1, data_set_0, initial_theta, lambda_param,
                                   calculate_classification_error_ridge, calculate_classification_error_ridge_gradient, calculate_classification_error)[1]

        accuracy_0 = calculate_predictor_accuracy(theta_0, data_set_1)
        accuracy_1 = calculate_predictor_accuracy(theta_1, data_set_0)

        average_accuracy = (accuracy_0 + accuracy_1) / 2.0
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            lambda_best = lambda_param

    return best_accuracy


def train_data(training_data, testing_data, theta, error_function, gradient_function):
    """Attempt to find a minimal value for theta such that the loss function is minimised.
    """
    # theta = scipy.optimize.fmin(error_function, x0=theta, args=tuple([training_data]), disp=False)
    theta = scipy.optimize.fmin_bfgs(error_function, x0=theta, args=tuple([training_data]), disp=False, fprime=gradient_function)

    error = error_function(theta, testing_data)
    return (error, theta)


def train_data_ridge(training_data, testing_data, theta, lambda_param, error_function, gradient_function, testing_function):
    """Attempt to find a minimal value for theta such that the loss function is minimised.
    """
    # theta = scipy.optimize.fmin(error_function, x0=theta, args=tuple([training_data, lambda_param]), disp=False)
    theta = scipy.optimize.fmin_bfgs(error_function, x0=theta, args=tuple([training_data, lambda_param]), disp=False, fprime=gradient_function)

    error = testing_function(theta, testing_data)
    return (error, theta)


def calculate_squared_error(theta, row_data):
    """Calculates the total squared error for all of the data, given theta.
    """
    total_squared_loss = 0

    for row in row_data:
        total_squared_loss += (numpy.dot(theta, row[-1]) - row[0])**2

    return total_squared_loss


def calculate_squared_error_gradient(theta, row_data):
    """Calculates the gradient of the squared error function.
    """
    gradient = [0] * len(row_data[0][-1])

    for row in row_data:
        gradient += numpy.dot(row[-1], numpy.dot(theta, row[-1]) - row[0]) * 2

    return gradient


def calculate_squared_error_ridge(theta, row_data, lambda_param):
    """Calculates the loss function for squared error with ridge regression.
    """
    loss_term = 0

    for row in row_data:
        loss_term += (numpy.dot(theta, row[-1]) - row[0])**2

    regularisation_term = sum(theta**2)

    return lambda_param * loss_term + (1 - lambda_param) * regularisation_term


def calculate_squared_error_ridge_gradient(theta, row_data, lambda_param):
    """Calculates the gradient for the loss function for squared error with ridge regression.
    """
    loss_term = [0] * len(row_data[0][-1])

    for row in row_data:
        loss_term += numpy.dot(row[-1], row[0] - (numpy.dot(theta, row[-1])))

    gradient = 2 * ((1 - lambda_param) * theta - lambda_param * loss_term)
    return gradient


def calculate_classification_error(theta, row_data):
    """Calculates the negative log likelihood so that it can be minimised.
    """
    result = calculate_log_likelihood(theta, row_data)
    return result


def calculate_classification_error_gradient(theta, row_data):
    """Calculates the gradient of the log likelihood function.
    """
    number_of_features = len(row_data[0][-1])
    gradient = [[0] * number_of_features] * number_of_classes

    for row in row_data:
        c = row[2]  # the class of this row of data
        for a in range(number_of_classes):
            # because the log likelihood was negated to allow minimisation, the
            # gradient function must be negated
            gradient[a] -= numpy.dot(row[-1], indicator(a, c) - probability_of_class(row, theta, c))

    result = []
    for c in range(len(gradient)):
        result += gradient[c].tolist()
    return result


def calculate_classification_error_ridge(theta, row_data, lambda_param):
    """Calculates the negative log likelihood with ridge regularisation so that it can be minimised.
    """
    error_term = calculate_log_likelihood(theta, row_data)
    regularisation_term = sum(theta**2)
    return (1 - lambda_param) * regularisation_term + lambda_param * error_term


def calculate_classification_error_ridge_gradient(theta, row_data, lambda_param):
    """Calculates the gradient of the log likelihood with ridge regularisation.
    """
    error_term = calculate_classification_error_gradient(theta, row_data)
    return numpy.dot(lambda_param, error_term) + 2 * (1 - lambda_param) * theta


def calculate_log_likelihood(theta, row_data):
    """Calculates the log likelihood of the data given theta.
    """
    total_sum = 0

    for row in row_data:
        features = row[-1]
        features_size = len(features)
        current_class = row[2]

        max_feature = max(features)
        logsumexp_term = scipy.misc.logsumexp([numpy.dot(features, get_theta_for_class(theta, i, features_size)) - max_feature for i in range(0, number_of_classes)])

        X_Theta_c_term = numpy.dot(features, get_theta_for_class(theta, current_class, features_size))

        total_sum += max_feature + logsumexp_term - X_Theta_c_term

    return total_sum


def indicator(a, c):
    """The indicator function for a class.
    """
    if a == c:
        return 1
    return 0


def calculate_predictor_accuracy(theta, row_data):
    """Calculates the accuracy of the classifier.
    """
    correct = 0
    incorrect = 0

    for row in row_data:
        predicted_class = predict_class(row, theta)
        if predicted_class == row[2]:
            correct += 1
        else:
            incorrect += 1

    return correct / float(correct + incorrect)


def predict_class(row, theta):
    """Finds the class with the highest probability that the data is likely to be.
    """
    max = 0
    max_class = -1
    probabilities = []

    for c in range(0, number_of_classes):
        result = probability_of_class(row, theta, c)
        probabilities.append(result)

        if result > max:
            max = result
            max_class = c

    return max_class


def probability_of_class(row, theta, c):
    """Calculates the probability that the data is of a particular class.
    """
    numerator_exponent = numpy.dot(row[-1], get_theta_for_class(theta, c, len(row[-1])))

    denominator_exponents = []
    for i in range(0, number_of_classes):
        denominator_exponents.append(numpy.dot(row[-1], get_theta_for_class(theta, i, len(row[-1]))))

    if numerator_exponent > 100 or max(denominator_exponents) > 100:
        return 0  # for large exponents, the fraction tends to 0

    numerator = math.exp(numerator_exponent)
    denominator = sum([math.exp(i) for i in denominator_exponents])

    return numerator / denominator


def get_theta_for_class(theta, c, features_size):
    """Gets the theta vector for a specific class.
    """
    start_index = features_size * c
    end_index = features_size * (c + 1)
    return theta[start_index:end_index]


def classify_day(today, yesterday):
    """Calculate the class of a day.
    """
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
    """Generate the features used for regression.
    """
    features = []

    # features += [1]  # constant
    # for row in rows:
    #     features += [float(row[0]) / 10000]
    #     features += [float(row[1])]
    features += [float(rows[-1][1])]  # yesterday's price

    return features


def collate_row_data_linear(raw_data):
    """Expands each row in the file and adds all of the necessary data to it in order to perform linear regression."
    """
    row_data = []

    for i in xrange(previous_days_count, len(raw_data)):
        row_data.append([])  # add a new row

        row_data[-1].append(float(raw_data[i][1]))  # add today's price to required data

        features = get_features(raw_data[i - previous_days_count:i])
        row_data[-1].append(features)  # append features to row

    # each row contains [today's price, [features]]
    return row_data


def collate_row_data_logistic(raw_data):
    """Expands each row in the file and adds all of the necessary data to it in order to perform logistic regression."
    """
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


