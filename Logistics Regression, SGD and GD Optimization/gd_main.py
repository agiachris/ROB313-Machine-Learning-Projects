import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import time
import random


__author__ = 'Christopher Agia (1003243509)'
__date__ = 'March 10, 2019'


# Useful Utility Functions
def compute_rmse(y_test, y_estimates):
    return np.sqrt(np.average((y_test-y_estimates)**2))


def compute_accuracy_ratio(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)


def l2_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=2)


# ---------------------- Question 1 ----------------------

def linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test):

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    # Train only with the first 1000 data points of training set
    x_total = x_total[:1000]
    y_total = y_total[:1000]

    # Create X matrix
    X = np.ones((len(x_total), len(x_total[0]) + 1))
    X[:, 1:] = x_total

    # Compute SVD
    U, S, Vh = np.linalg.svd(X)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([len(x_total)-len(S), len(S)])
    sig_inv = np.linalg.pinv(np.vstack([sig, filler]))

    # Compute weights and predictions
    w = np.dot(Vh.T, np.dot(sig_inv, np.dot(U.T, y_total)))

    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    predictions = np.dot(X_test, w)

    result = compute_rmse(y_test, predictions)
    return result, w


def gd(x_train, x_test, y_train, y_test, rates, opt_rmse, model):

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # One percent upper bound on optimal test rmse (from SVD LM)
    opt_rmse = opt_rmse*1.01

    # Create X matrix (Used for all rates)
    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train

    # Create X_test matrix for test predictions
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test

    # Return Variables
    times = list()
    w_opts = list()
    test_rmses = list()
    loss = {}

    for rate in rates:

        # Initialize minimizer
        w = np.zeros((33, 1))

        loss[rate] = list()
        timed_in = False

        # Start Gradient Descent Process
        total_time = 0
        tic = time.time()
        for iteration in range(1000):

            # Make predictions
            predictions = np.dot(X, w)

            if model == 'SGD':
                # Compute Mini-Batch (1) Gradient
                i = random.randint(0, len(predictions)-1)
                grad_L = 2 * (predictions[i] - y_train[i])*np.insert(x_train[i], 0, 1)

            elif model == 'GD':
                # Compute Full-Batch Gradient
                grad_L = np.insert(np.zeros(np.shape(x_train[0])), 0, 0)
                for i in range(len(predictions)):
                    grad_L += (predictions[i]-y_train[i])*np.insert(x_train[i], 0, 1)

                # Normalize
                grad_L = 2*grad_L/len(predictions)

            # Reshape
            grad_L = grad_L.reshape((33, 1))

            # Update weights
            w = np.add(w, -rate*grad_L)

            # Halt Timer (computations underneath not part of GD or SGD)
            toc = time.time()
            total_time += toc - tic

            # Calculate Full-Batch Loss
            L = 0
            for i in range(len(predictions)):
                L += (predictions[i]-y_train[i])**2
            loss[rate].append(L/len(predictions))

            # Prediction on test set
            test_estimates = np.dot(X_test, w)
            test_error = compute_rmse(test_estimates, y_test)
            if test_error <= opt_rmse and not timed_in:
                times.append((total_time, iteration + 1))
                timed_in = True

            # Restart Timer
            tic = time.time()

        # For each learning rate record
        # (1) total time to 5000 iterations or within 1% optimal test RMSE
        # (2) final minimizer after 5000 iterations
        # (3) final Test RMSE after 5000 iterations

        if not timed_in:
            # Record Training Time
            toc = time.time()
            total_time += toc - tic
            times.append((total_time, 1000))

        test_estimates = np.dot(X_test, w)
        test_error = compute_rmse(test_estimates, y_test)
        test_rmses.append(test_error)

        # Record Minimizer
        w_opts.append(w)

    if model == 'GD':
        min_test_rmse = min(test_rmses)
        min_w = w_opts[test_rmses.index(min_test_rmse)]
        min_it = np.inf
        for i in range(len(times)):
            if times[i][1] < min_it:
                min_it = times[i][1]
                preferred_rate = rates[i]

    else:
        min_test_rmse = min(test_rmses)
        min_w = w_opts[test_rmses.index(min_test_rmse)]
        preferred_rate = rates[test_rmses.index(min_test_rmse)]

    return loss, times, min_test_rmse, min_w, preferred_rate


def generate_lossplots(x_train, y_train, losses, opt_w, model):

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train

    opt_pred = np.dot(X, opt_w)
    L_opt = 0
    for i in range(len(opt_pred)):
        L_opt += (opt_pred[i]-y_train[i])**2
    L_opt = L_opt / len(opt_pred)

    x_axis = list(range(1000))
    L = [L_opt] * 1000

    if model == 'GD':
        plt.figure(1)
        plt.title('GD Full-Batch Loss vs Iteration')
        plt.plot(x_axis, L, '--k', label='Optimal Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Full-Batch Loss')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='Loss for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.savefig('GD_loss.png')

        plt.plot(x_axis, L, '--k', label='Optimal Loss')
    elif model == 'SGD':
        plt.figure(2)
        plt.title('SGD Full-Batch Loss vs Iteration')
        plt.plot(x_axis, L, '--k', label='Optimal Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Full-Batch Loss')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='Loss for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.savefig('SGD_loss.png')


# ---------------------- Question 2 ----------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(estimates, actual):
    total = 0
    for i in range(len(estimates)):
        total += actual[i]*np.log(sigmoid(estimates[i])) + (1-actual[i])*np.log(1 - sigmoid(estimates[i]))
    return total


def log_reg_GD(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, model):

    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # Create X matrix (Used for all rates)
    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train

    # Create X_test matrix for test predictions
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test

    # Return Variables
    times = list()
    test_accuracies = list()
    test_logs = list()
    neg_log = {}

    for rate in rates:

        # Initialize minimizer
        w = np.zeros(np.shape(X[0, :]))

        neg_log[rate] = list()

        # Start Gradient Descent Process
        total_time = 0
        tic = time.time()
        for iteration in range(5000):

            # LM Estimates (on training set)
            estimates = np.dot(X, w)
            estimates = estimates.reshape(np.shape(y_train))

            if model == 'SGD':
                # Compute Mini-Batch (1) Gradient
                i = random.randint(0, len(y_train)-1)
                grad_L = (y_train[i] - sigmoid(estimates[i])) * X[i, :]

            elif model == 'GD':
                # Compute Full-Batch Gradient
                grad_L = np.zeros(np.shape(w))
                for i in range(len(y_train)):
                    grad_L += (y_train[i] - sigmoid(estimates[i])) * X[i, :]

            # Update weights
            w = np.add(w, rate*grad_L)

            # Halt Timer (computations underneath not part of GD or SGD)
            toc = time.time()
            total_time += toc - tic

            # Calculate Full-Batch Log-Likelihood
            L = log_likelihood(estimates, y_train)
            neg_log[rate].append(-L)

            # Restart Timer
            tic = time.time()

        # Record Final Time
        toc = time.time()
        total_time += toc - tic
        times.append(total_time)

        # Allocate Space and make classifications, record test accuracy ratio
        test_estimates = np.dot(X_test, w)
        test_estimates = test_estimates.reshape(np.shape(y_test))
        predictions = np.zeros(np.shape(y_test))
        for i in range(len(predictions)):
            p = sigmoid(test_estimates[i])
            if p > 1/2:
                predictions[i] = 1
            elif p < 1/2:
                predictions[i] = 0
            else:
                predictions[i] = -1

        # Append Test Accuracy and Test Log-likelihood
        test_accuracies.append(compute_accuracy_ratio(y_test, predictions))
        test_logs.append(log_likelihood(test_estimates, y_test))

    # Final Recordings (Best Accuracy, Log-likelihood, Preferred Rates)
    best_accuracy = max(test_accuracies)
    test_log = min(test_logs)
    min_rates = list()
    min_rates.append(rates[test_accuracies.index(best_accuracy)])
    min_rates.append(rates[test_logs.index(test_log)])

    return neg_log, times, best_accuracy, min_rates, test_log


def log_reg_plots(losses, model):

    x_axis = list(range(5000))
    if model == 'GD':
        plt.figure(3)
        plt.title('GD Full-Batch Negative Log-Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('(-)Log-L')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='(-)Log-L for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.savefig('GD_log_l.png')

    elif model == 'SGD':
        plt.figure(4)
        plt.title('SGD Full-Batch Negative Log-Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('(-)Log-L')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='(-)Log-L for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.savefig('SGD_log_l.png')


# ---------------------- Main Block ----------------------

if __name__ == '__main__':

    # Only used to acquire the optimal minimizer from Assignment 1
    LinearModel = False
    # Set True for Question 1
    Q1 = False
    # Set True for Question 2
    Q2 = False

    # All Dataset Names
    all_datasets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    regression_sets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    classification_sets = ['iris', 'mnist_small']
    print('')

    # --------------------- Linear Model ---------------------

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    test_rmse, opt_w = linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test)

    if LinearModel:
        print('----------------- Overall Results for Linear Model ------------------')
        print('')
        print('Test RMSE: ' + str(test_rmse))
        print('Optimal Minimizer (through SVD):')
        print(opt_w)

    # ---------------------- Question 1 ----------------------

    if Q1:
        print('------------------ Overall Results for Question 1 -------------------')
        print('')

        rates = [0.0001, 0.001, 0.01, 0.1]
        gd_loss, conv_times_gd, test_rmse_gd, opt_w_gd, pref_rGD = gd(x_train, x_test, y_train, y_test, rates, test_rmse, 'GD')
        generate_lossplots(x_train, y_train, gd_loss, opt_w, 'GD')

        rates = [0.0001, 0.001, 0.01]
        sgd_loss, conv_times_sgd, test_rmse_sgd, opt_w_sgd, pref_rSGD = gd(x_train, x_test, y_train, y_test, rates, test_rmse,'SGD')
        generate_lossplots(x_train, y_train, sgd_loss, opt_w, 'SGD')

        print('-- Full-Batch GD --')
        print('Test RMSE (GD): ' + str(test_rmse_gd))
        print('Convergence Times (GD): ' + str(conv_times_gd))
        print('Preferred Rate (GD): ' + str(pref_rGD))
        print('Optimal Minimizer (Full-Batch):')
        print(opt_w_gd)
        print('')

        print('-- Stochastic GD --')
        print('Test RMSE (SGD): ' + str(test_rmse_sgd))
        print('Convergence Times (SGD): ' + str(conv_times_sgd))
        print('Preferred Rate (SGD): ' + str(pref_rSGD))
        print('Optimal Minimizer (SGD):')
        print(opt_w_sgd)
        print('')

    # ---------------------- Question 2 ----------------------

    if Q2:
        print('------------------ Overall Results for Question 2 -------------------')
        print('')

        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
        y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

        y_train = np.asarray(y_train, int)
        y_valid = np.asarray(y_valid, int)
        y_test = np.asarray(y_test, int)

        rates = [0.0001, 0.001, 0.01]
        log_gd, times_gd, ratio_gd, pref_rGD, test_log_gd = log_reg_GD(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, 'GD')
        log_reg_plots(log_gd, 'GD')

        rates = [0.0001, 0.001, 0.01]
        log_sgd, times_sgd, ratio_sgd, pref_rSGD, test_log_sgd = log_reg_GD(x_train, x_valid, x_test, y_train, y_valid, y_test, rates, 'SGD')
        log_reg_plots(log_sgd, 'SGD')

        print('-- Full-Batch GD --')
        print('Test Accuracy Ratio (GD): ' + str(ratio_gd))
        print('Test Log-likelihood (GD: ' + str(test_log_gd))
        print('Convergence Times (GD): ' + str(times_gd))
        print('Preferred Rate (GD-Ratio): ' + str(pref_rGD[0]))
        print('Preferred Rate (GD-Log): ' + str(pref_rGD[1]))
        print('')

        print('-- Stochastic GD --')
        print('Test Accuracy Ratio (SGD): ' + str(ratio_sgd))
        print('Test Log-likelihood (SGD): ' + str(test_log_sgd))
        print('Convergence Times (SGD): ' + str(times_sgd))
        print('Preferred Rate (SGD-Ratio): ' + str(pref_rSGD[0]))
        print('Preferred Rate (SGD-Log): ' + str(pref_rSGD[1]))
        print('')
