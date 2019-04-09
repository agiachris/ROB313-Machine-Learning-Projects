from a4_mod import *
from matplotlib import pyplot as plt
from data_utils import load_dataset
from data_utils import plot_digit
import time
import random


__author__ = 'Christopher Agia (1003243509)'
__date__ = 'March 10, 2019'


# Useful Utility Functions
def compute_rmse(y_test, y_estimates):
    return np.sqrt(np.average((y_test-y_estimates)**2))


def accuracy_ratio(W1, W2, W3, b1, b2, b3, x, y):
    Fhat = np.exp(forward_pass(W1, W2, W3, b1, b2, b3, x))
    Fhat = np.argmax(Fhat, axis=1)
    y = np.argmax(y, axis=1)
    return (Fhat == y).sum() / len(y)


def l2_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=2)


def update_weights(w, grad_w, learning_rate, dir):
    return w - dir*learning_rate*grad_w


def gen_plot(i, losses, len, name, filename):
    x_axis = list(range(len))
    if name == 'Train':
        normalization = 10000
    if name == 'Valid':
        normalization = 1000

    for rate in losses:
        losses[rate] = [x / normalization for x in losses[rate]]

    # Generate plots
    plt.figure(i)
    plt.title('SGD(250) Full-Batch Normalized (-)Log-Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('(-)Log-L')
    for rate in losses:
        plt.plot(x_axis, losses[rate], label=(name+' (-)LL n=' + str(rate)))
    plt.legend(loc='upper right')
    plt.savefig(filename + '.png')


def plot_digit_mod(x, i, j, neuron=False):
    """ plots a provided MNIST digit """
    assert np.size(x) == 784
    plt.imshow(x.reshape((28, 28)), interpolation='none', aspect='equal', cmap='gray')
    if neuron:
        plt.savefig('W1[neuron_' + str(i+1) + '].png')
    else:
        plt.savefig('test_digit_' + str(i) + '_rank_' + str(j) + '.png')


# Questions 3 and 4
def nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, M, B, len, rates=[0.0001, 0.001], test=False,\
           vis=False, digit=False):

    # initialize the weights and biases of the network (xavier for weights, zero for biases
    W1 = np.random.randn(M, 784) / np.sqrt(784)
    W2 = np.random.randn(M, M) / np.sqrt(M)
    W3 = np.random.randn(10, M) / np.sqrt(M)
    b1 = np.zeros((M, 1))
    b2 = np.zeros((M, 1))
    b3 = np.zeros((10, 1))

    # Parameters to track
    neg_ll_train = {}
    neg_ll_valid = {}

    for rate in rates:

        neg_ll_train[rate] = list()
        neg_ll_valid[rate] = list()

        # holding the minimum validation log-likelihood
        min_ll_valid = np.inf
        min_ll_valid_it = 0

        for i in range(len):

            if not vis:
                # compute full-batch negative log-likelihood for training and validation set

                nll_valid_fb = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
                neg_ll_valid[rate].append(nll_valid_fb)

                # Track minimum log-likelihood for validation set (and corresponding weights)
                if nll_valid_fb < min_ll_valid:
                    min_ll_valid = nll_valid_fb
                    min_ll_valid_it = i
                    W1_opt = W1
                    W2_opt = W2
                    W3_opt = W3
                    B1_opt = b1
                    B2_opt = b2
                    B3_opt = b3

            # compute list of 250 random integers (mini-batch indices)
            idx = np.random.choice(x_train.shape[0], size=B, replace=False)
            mini_batch_x = x_train[idx, :]
            mini_batch_y = y_train[idx, :]

            # compute log-likelihood and corresponding gradients over the mini-batch
            (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
                nll_gradients(W1, W2, W3, b1, b2, b3, mini_batch_x, mini_batch_y)

            if not test and not vis:
                neg_ll_train[rate].append(nll/250*10000)

            # update weights
            W1 = update_weights(W1, W1_grad, rate, 1)
            W2 = update_weights(W2, W2_grad, rate, 1)
            W3 = update_weights(W3, W3_grad, rate, 1)
            b1 = update_weights(b1, b1_grad, rate, 1)
            b2 = update_weights(b2, b2_grad, rate, 1)
            b3 = update_weights(b3, b3_grad, rate, 1)

        if not vis and not digit:
            # Display results
            print('Results for ' + str(M) + ' neurons with learning rate: ' + str(rate))
            if not test:
                print('Train (-)LL: ' + str(neg_ll_train[rate][-1]))
            print('Valid (-)LL: ' + str(neg_ll_valid[rate][-1]))
            print('Minimum Valid (-)LL: ' + str(min_ll_valid) + ' at iteration ' + str(min_ll_valid_it + 1))

            # Acquire optimal validation set and test set log-likelihood and accuracy ratio
            if test:
                valid_ratio = accuracy_ratio(W1_opt, W2_opt, W3_opt, B1_opt, B2_opt, B3_opt, x_valid, y_valid)
                test_ratio = accuracy_ratio(W1_opt, W2_opt, W3_opt, B1_opt, B2_opt, B3_opt, x_test, y_test)
                test_nll = negative_log_likelihood(W1_opt, W2_opt, W3_opt, B1_opt, B2_opt, B3_opt, x_test, y_test)
                print('Optimal Valid Ratio: ' + str(valid_ratio))
                print('Optimal Test (-)LL: ' + str(test_nll) + ' at iteration ' + str(min_ll_valid_it + 1))
                print('Optimal Test Ratio: ' + str(test_ratio))

        if digit:
            Fhat = np.max(np.exp(forward_pass(W1_opt, W2_opt, W3_opt, B1_opt, B2_opt, B3_opt, x_test)), axis=1)
            sorted_ind = np.argsort(Fhat)
            sorted_test_set = x_test[sorted_ind]

        print('')

    if digit:
        return sorted_ind, sorted_test_set

    if not vis:
        if not test:
            return neg_ll_train, neg_ll_valid

        return neg_ll_valid

    else:
        seen = list()
        for i in range(17):
            j = np.random.randint(M)
            if j not in seen:
                seen.append(j)
                plot_digit_mod(W1[j], j, 0, neuron=True)
            else:
                i -= 1


if __name__ == '__main__':

    # Set True for Question 3
    Q3 = False
    # Set True for Question 4
    Q4 = False
    # Set True for Question 5
    Q5 = False
    # Set True for Question 6
    Q6 = False

    # All Dataset Names
    all_datasets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    regression_sets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    classification_sets = ['iris', 'mnist_small']
    print('')

    # Acquire and Transform Data (zero mean, unit variance)
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')

    # Question 3
    if Q3:
        print('----- Results for Q3 -----')
        print('')
        train_nll, valid_nll = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 4000, [0.0001])
        gen_plot(1, train_nll, 4000, 'Train', 'mb250_nn100_0.0001')
        gen_plot(1, valid_nll, 4000, 'Valid', 'mb250_nn100_0.0001')
        train_nll, valid_nll = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 1000, [0.001])
        gen_plot(2, train_nll, 1000, 'Train', 'mb250_nn100_0.001')
        gen_plot(2, valid_nll, 1000, 'Valid', 'mb250_nn100_0.001')

    # Question 4
    if Q4:
        print('----- Results for Q4 -----')
        print('')
        valid_nll_s = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 10, 250, 2000, test=True)
        valid_nll_m = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 90, 250, 2000, test=True)
        valid_nll_l = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 180, 250, 2000, test=True)

    # Question 5
    if Q5:
        nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 1000, [0.001], vis=True)

    # Question 6
    if Q6:
        ind, sorted_test_set = nn_sgd(x_train, x_valid, x_test, y_train, y_valid, y_test, 100, 250, 1000, [0.001],\
                                      test=True, digit=True)
        for i in range(20):
            plot_digit_mod(sorted_test_set[i], ind[i], i)
