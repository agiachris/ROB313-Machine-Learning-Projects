import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import heapq
from sklearn import neighbors

__author__ = 'Christopher Agia (1003243509)'
__date__ = 'February 12, 2019'


# Root Mean Squared Error Function
def rmse(y_test, y_estimates):
    return np.sqrt(np.average((y_test-y_estimates)**2))

# Norm Utility Functions
def l1_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=1)


def l2_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=2)


def linf_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=np.inf)


def ff_regression(x_train, x_valid, y_train, y_valid, distance_functions, k_list=None):

    assert((len(x_train)+len(x_valid)) == (len(y_train)+len(y_valid)))

    rmse_vals = {}
    error = []

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])
    np.random.seed(5)
    np.random.shuffle(x_total)
    np.random.seed(5)
    np.random.shuffle(y_total)

    ff_length = len(x_total)//5

    #  Trial K values to try
    if not k_list:
        k_list = list(range(0, 30))

    # Iterate over 5-folds
    for i in range(5):

        # Obtain validation and train set associated with fold i
        y_valid = y_total[i * ff_length:(i + 1) * ff_length]
        y_train = np.vstack([y_total[:i * ff_length], y_total[(i + 1) * ff_length:]])
        x_valid = x_total[i * ff_length:(i + 1) * ff_length]
        x_train = np.vstack([x_total[:i * ff_length], x_total[(i + 1) * ff_length:]])

        # Loop over distance functions (e.g. l1_norm, l2_norm, linf_norm)
        for func in distance_functions:
            y_est = {}
            # Compute distances according to distance function for one validation point
            for j in range(ff_length):
                d = []
                for t in range(len(x_train)):
                    d.append((func(x_train[t], x_valid[j]), y_train[t]))

                # Sort distances to take the nearest k-values
                d.sort(key=lambda x: x[0])

                # Compute y_estimate for validation point j
                for k in k_list:
                    y = 0
                    for elem in d[:k+1]:
                        y += elem[1]

                    if k not in y_est:
                        y_est[k] = []

                    y_est[k].append(y/(k+1))

            # Compute root mean squared error for each k-value
            for k in k_list:
                if (func, k) not in rmse_vals:
                    rmse_vals[(func, k)] = []
                rmse_vals[(func, k)].append(rmse(y_valid, y_est[k]))


    # Error will contain 30 rmse-error values as per k for each function
    for func, k in rmse_vals:
        ff_error = sum(rmse_vals[(func, k)]) / 5
        error.append((k+1, func, ff_error))

    return error


def ff_regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k, func, plot=False):

    # Create total training set (no validation)
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    # Predictions for each test data point will be stored in this list
    predictions = []

    # Iterate over test points
    for elem in x_test:

        # Compute distances between all training points and test point
        d = []
        for i in range(len(x_total)):
            d.append((func(elem, x_total[i]), y_total[i]))

        # Sort in terms of increasing distance
        d.sort(key=lambda x: x[0])

        # Average over k nearest y values
        y_est = 0
        for item in d[:k]:
            y_est += item[1]
        avg = y_est/k

        # Append to predictions
        predictions.append(avg)

    test_error = rmse(y_test, predictions)

    if plot:
        # Predictions on test set
        plt.figure(2)
        plt.plot(x_test, y_test, '-bo', label='True Values')
        plt.plot(x_test, predictions, '-ro', label='Predicted Values')
        plt.title('Test Predictions for Mauna Loa')
        plt.xlabel('x_test')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig('mauna_loa_prediction.png')

    return test_error


def one_fold_classification(x_train, y_train, x_valid, y_valid, distance_functions, k_list=None):

    assert ((len(x_train) + len(x_valid)) == (len(y_train) + len(y_valid)))

    tally = {}

    if not k_list:
        k_list = list(range(0, 30))

    # Loop over distance functions (e.g. l1_norm, l2_norm, linf_norm)
    for func in distance_functions:

        # Compute distances according to distance function for one validation point
        for j in range(len(x_valid)):
            d = []
            for t in range(len(x_train)):
                d.append((func(x_train[t], x_valid[j]), y_train[t]))

            # Sort distances to take the nearest k-values
            d.sort(key=lambda x: x[0])

            classes = {}
            # Identify k-nearest neighbours for x_valid[j]
            for k in k_list:
                classes[k] = []
                for elem in d[:k + 1]:
                    classes[k].append(elem[1])

            for k in k_list:
                occurance = {}
                for point in classes[k]:
                    if str(point) not in occurance:
                        occurance[str(point)] = (point, 0)
                    occurance[str(point)] = (point, occurance[str(point)][1] + 1)

                occur_list = list(occurance.values())
                occur_list.sort(key=lambda x: x[1], reverse=True)

                if np.all(occur_list[0][0] == y_valid[j]):
                    if (k + 1, func) not in tally:
                        tally[(k + 1, func)] = 0
                    tally[(k + 1, func)] += 1

    results = []
    for k, func in tally:
        ratio = tally[(k, func)]/len(y_valid)
        result = (k, func, ratio)
        results.append(result)

    return results


def classification_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k, func):

    # Create total training set (no validation)
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    tally = 0
    # Iterate over test points
    for i in range(len(x_test)):

        # Compute distances between all training points and test point
        d = []
        for j in range(len(x_total)):
            d.append((func(x_test[i], x_total[j]), y_total[j]))

        # Sort in terms of increasing distance
        d.sort(key=lambda x: x[0])

        occurance = {}
        for item in d[: k]:
            if str(item[1]) not in occurance:
                occurance[str(item[1])] = (item[1], 0)
            occurance[str(item[1])] = (item[1], occurance[str(item[1])][1] + 1)

        occur_list = list(occurance.values())
        occur_list.sort(key=lambda x: x[1], reverse=True)

        if np.all(occur_list[0][0] == y_test[i]):
            tally += 1

    return tally/len(x_test)


def run_model(model_type, dataset):

    functions = [l1_norm, l2_norm, linf_norm]

    if model_type == 'regression':

        if dataset == 'rosenbrock':
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, n_train=1000, d=2)
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)

        if dataset == 'mauna_loa':
            result = ff_regression(x_train, x_valid, y_train, y_valid, [l2_norm])

            # Result = (k, Function, RMSE Error)
            result.sort(key=lambda x: x[0])

            k_vals = []
            error_vals = []
            for k, func, error in result:
                k_vals.append(k)
                error_vals.append(error)

            plt.figure(1)
            plt.plot(k_vals, error_vals, '-bo')
            plt.xlabel('k')
            plt.ylabel('Average 5-Fold RMSE')
            plt.title(dataset + ' 5-Fold l2 Loss')
            plt.savefig(dataset +'l2_loss.png')

            result.sort(key=lambda x: x[2])
            k_min = result[0][0]
            func = result[0][1]
            test_error = ff_regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_min, func, plot=True)

            print('----------------------------------------------')
            print('Results for Mauna Loa with L2 Norm :')
            print('')
            print('Optimal k: ' + str(k_min))
            print('Optimal Distance Metric: ' + str(func))
            print('Five fold RMSE: ' + str(result[0][2]))
            print('Test RMSE: ' + str(test_error))
            print('')

        result = ff_regression(x_train, x_valid, y_train, y_valid, functions)
        result.sort(key=lambda x: x[2])
        k_min = result[0][0]
        func = result[0][1]
        test_error = ff_regression_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_min, func)

        return result[0][0], result[0][1], result[0][2], test_error

    elif model_type == 'classification':

        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
        one_fold_result = one_fold_classification(x_train, y_train, x_valid, y_valid, functions)
        one_fold_result.sort(key=lambda x: x[2], reverse=True)
        # ff_result = (k, func, ratio)
        k_min = one_fold_result[0][0]
        func = one_fold_result[0][1]
        test_ratio = classification_test(x_train, x_valid, x_test, y_train, y_valid, y_test, k_min, func)

        return k_min, func, one_fold_result[0][2], test_ratio

    return 0


def regression_performance(x_total, x_test, y_total, y_test, k, f, method):

    start = time.time()

    # Predictions for each test data point will be stored in this list
    predictions = []

    # Brute-force Double For Loop Method
    if method == 'a':

        # Iterate over test points
        for elem in x_test:

            # Compute distances between all training points and test point
            d = []
            for j in range(len(x_total)):
                d.append((f(elem, x_total[j]), y_total[j]))

            # Sort in terms of increasing distance
            d.sort(key=lambda x: x[0])

            # Average over k nearest y values
            y_est = 0
            for item in d[:k]:
                y_est += item[1]
            avg = y_est/k

            # Append to predictions
            predictions.append(avg)

        test_error = rmse(y_test, predictions)

    # Half-Vectorization Method
    elif method == 'b':

        for j in range(len(x_test)):
            d = np.sqrt(np.sum(np.square(x_total - x_test[j]), axis=1))
            k_nb = heapq.nsmallest(k, range(len(d)), d.take)
            predictions.append(np.array([(np.average(np.take(y_total, k_nb)))]))

        test_error = rmse(y_test, predictions)

    # Full Vectorization Method
    elif method == 'c':

        d = np.sqrt(-2 * np.dot(x_test, x_total.T) + np.sum(x_total ** 2, axis=1) + np.sum(x_test ** 2, axis=1)[:, np.newaxis])
        k_nb = np.argpartition(d, kth=k, axis=1)[:, : k]
        predictions = np.sum(y_total[k_nb], axis=1) / k

        test_error = rmse(y_test, predictions)

    # K-D Tree Method
    elif method == 'd':

        kdt = neighbors.KDTree(x_total)
        d, k_nb = kdt.query(x_test, k=k)
        predictions = np.sum(y_total[k_nb], axis=1) / k

        test_error = rmse(y_test, predictions)

    runtime = time.time() - start
    return runtime, test_error


def linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, model_type):

    if model_type == 'regression':

        x_total = np.vstack([x_train, x_valid])
        y_total = np.vstack([y_train, y_valid])

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

        result = rmse(y_test, predictions)

    elif model_type == 'classification':

        x_total = np.vstack([x_train, x_valid])
        y_total = np.vstack([y_train, y_valid])

        # Expand X matrix
        X = np.ones([len(x_total), len(x_total[0]) + 1])
        X[:, 1:] = x_total

        # Convert to integer
        #y_test = 1 * y_test

        # Perform SVD
        U, S, Vh = np.linalg.svd(X)

        # Expand Sigma Matrix
        sig = np.diag(S)
        filler = np.zeros([len(x_total) - len(S), len(S)])
        sig_inv = np.linalg.pinv(np.vstack([sig, filler]))

        # Compute weights
        w = np.dot(Vh.T, np.dot(sig_inv, np.dot(U.T, y_total)))

        # Create Test Matrix
        X_test = np.ones([len(x_test), len(x_test[0]) + 1])
        X_test[:, 1:] = x_test

        # find prediction accuracy
        predictions = np.argmax(np.dot(X_test, w), axis=1)
        y_test = np.argmax(1 * y_test, axis=1)

        result = (predictions == y_test).sum() / len(y_test)

    return result


if __name__ == '__main__':
    # All Dataset Names
    all_datasets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    regression_sets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    classification_sets = ['iris', 'mnist_small']

    # ---------------------- Question 1 ----------------------

    # print('------------------ Overall Results for Question 1 -------------------')
    # print('')
    # for d_set in regression_sets:
    #     k_min, metric_min, ff_rmse, test_rmse = run_model('regression', d_set)
    #     print('----------------------------------------------')
    #     print('Results for ' + d_set + ' :')
    #     print('')
    #     print('Optimal k: ' + str(k_min))
    #     print('Optimal Distance Metric: ' + str(metric_min))
    #     print('Five fold RMSE: ' + str(ff_rmse))
    #     print('Test RMSE: ' + str(test_rmse))
    #     print('')


    # ---------------------- Question 2 ----------------------

    # print('------------------ Overall Results for Question 2 -------------------')
    # print('')
    # for d_set in classification_sets:
    #     k_min, metric_min, max_ratio, test_ratio = run_model('classification', d_set)
    #     print('----------------------------------------------')
    #     print('Results for ' + d_set + ' :')
    #     print('')
    #     print('Optimal k: ' + str(k_min))
    #     print('Optimal Distance Metric: ' + str(metric_min))
    #     print('Validation Ratio: ' + str(max_ratio))
    #     print('Test Ratio: ' + str(test_ratio))
    #     print('')


    # ---------------------- Question 3 ----------------------

    # result_table = {}
    # result_table['Double Loop'] = []
    # result_table['Half Vectorized'] = []
    # result_table['Full Vectorized'] = []
    # result_table['k-d Tree'] = []
    #
    # for ind_d in range(2, 10):
    #
    #     x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=ind_d)
    #     x_total = np.vstack([x_train, x_valid])
    #     y_total = np.vstack([y_train, y_valid])
    #
    #     run_time, test_rmse = regression_performance(x_total, x_test, y_total, y_test, 5, l2_norm, 'a')
    #     result_table['Double Loop'].append((ind_d, run_time, test_rmse))
    #
    #     run_time, test_rmse = regression_performance(x_total, x_test, y_total, y_test, 5, l2_norm, 'b')
    #     result_table['Half Vectorized'].append((ind_d, run_time, test_rmse))
    #
    #     run_time, test_rmse = regression_performance(x_total, x_test, y_total, y_test, 5, l2_norm, 'c')
    #     result_table['Full Vectorized'].append((ind_d, run_time, test_rmse))
    #
    #     run_time, test_rmse = regression_performance(x_total, x_test, y_total, y_test, 5, l2_norm, 'd')
    #     result_table['k-d Tree'].append((ind_d, run_time, test_rmse))
    #
    # print('------------------ Overall Results for Question 3 -------------------')
    # print('')
    #
    # m = list(result_table.keys())
    # for r in range(8):
    #     print('--- Results for d = ' + str(r+2) + ' ---')
    #     print('')
    #     print('Times-- ' + m[0] + ': ' + str(result_table[m[0]][r][1]) + '  ' + m[1] + ': ' + str(result_table[m[1]][r][1]))
    #     print('        ' + m[2] + ': ' + str(result_table[m[2]][r][1]) + '  ' + m[3] + ': ' + str(result_table[m[3]][r][1]))
    #     print('RMSEs-- ' + m[0] + ': ' + str(result_table[m[0]][r][2]) + '  ' + m[1] + ': ' + str(result_table[m[1]][r][2]))
    #     print('        ' + m[2] + ': ' + str(result_table[m[2]][r][2]) + '  ' + m[3] + ': ' + str(result_table[m[3]][r][2]))
    #     print('')
    #
    # d_arr = list(range(2, 10))
    # plt.figure(3)
    # plt.title('Runtimes as a function of d')
    # plt.xlabel('d')
    # plt.ylabel('Time (s)')
    # plt.figure(4)
    # plt.title('RMSE as a function of d')
    # plt.xlabel('d')
    # plt.ylabel('RMSE')
    # count = 0
    #
    # for m in result_table:
    #
    #     if count == 0:
    #         tab = '-b'
    #     elif count == 1:
    #         tab = '-g'
    #     elif count == 2:
    #         tab = '-r'
    #     else:
    #         tab = '-m'
    #     count += 1
    #
    #     runtimes = []
    #     rmses = []
    #     for i in range(len(result_table[m])):
    #         runtimes.append(result_table[m][i][1])
    #         rmses.append(result_table[m][i][2])
    #
    #     plt.figure(3)
    #     plt.plot(d_arr, runtimes, tab, label=m)
    #
    #     plt.figure(4)
    #     plt.plot(d_arr, rmses, tab, label=m)
    #
    # plt.figure(3)
    # plt.legend(loc='upper right')
    # plt.savefig('runtimes_vs_d.png')
    #
    # plt.figure(4)
    # plt.legend(loc='upper right')
    # plt.savefig('rmses_vs_d.png')

    # ---------------------- Question 4 ----------------------

    # print('------------------ Overall Results for Question 4 -------------------')
    # print('')
    # for d_set in regression_sets:
    #
    #     if d_set == 'rosenbrock':
    #         x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(d_set, n_train=1000, d=2)
    #     else:
    #         x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(d_set)
    #
    #     final_rmse = linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, 'regression')
    #     print('Test RMSE for ' + d_set + ': ' + str(final_rmse))
    #
    #
    # for d_set in classification_sets:
    #
    #     x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(d_set)
    #
    #     final_ratio = linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test, 'classification')
    #     print('Test Ratio for ' + d_set + ': ' + str(final_ratio))
