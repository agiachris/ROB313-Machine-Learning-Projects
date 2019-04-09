import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset


__author__ = 'Christopher Agia (1003243509)'
__date__ = 'February 18, 2019'


# Useful Utility Functions
def compute_rmse(y_test, y_estimates):
    return np.sqrt(np.average((y_test-y_estimates)**2))


def l2_norm(x1, x2):
    return np.linalg.norm([x1-x2], ord=2)


# Kernel Functions and Custom Vector Function
def custom_kernel(x0, x1):
    # Customer kernel incorporates a polynomial expansion term and product of two sinusoidal terms
    return (1+x0*x1)**2 + x0*x1*math.cos(2*math.pi/0.0565*(x0-x1))


def custom_vector(x):
    freq = 2*math.pi/0.0565
    row = list()
    row.append(1)
    row.append(math.sqrt(2)*x)
    row.append(x**2)
    row.append(x * math.sin(freq * x))
    row.append(x * math.cos(freq * x))
    return np.array(row)


# ---------------------- Question 1 ----------------------

def glm_validation(x_train, x_valid, y_train, y_valid, l_vals=None):

    # Regularization Parameters ranging from 0-30
    if not l_vals:
        l_vals = list(range(0, 31))

    # Create and Populate the PHI Matrix
    shape = (len(x_train), 5)
    phi = np.empty(shape)
    for i in range(len(x_train)):
        phi[i, :] = custom_vector(x_train[i])

    # Create validation phi matrix
    shape = (len(x_valid), 5)
    phi_valid = np.empty(shape)
    for i in range(len(x_valid)):
        phi_valid[i, :] = custom_vector(x_valid[i])

    # Compute SVD
    U, S, Vh = np.linalg.svd(phi)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([len(x_train) - len(S), len(S)])
    sig = np.vstack([sig, filler])

    # Compute weights and predictions with varying lambda values
    min_rmse = np.inf
    for l_val in l_vals:
        temp0 = np.dot(sig.T, sig)
        temp1 = np.linalg.inv(temp0 + l_val*np.eye(len(temp0)))

        w = np.dot(Vh.T, np.dot(temp1, np.dot(sig.T, np.dot(U.T, y_train))))

        predictions = np.dot(phi_valid, w)
        rmse_val = compute_rmse(y_valid, predictions)

        if rmse_val < min_rmse:
            min_rmse = rmse_val
            l_min = l_val

    return l_min


def glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l_val):

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    # Create and Populate the training PHI Matrix
    shape = (len(x_total), 5)
    phi = np.empty(shape)
    for i in range(len(x_total)):
        phi[i, :] = custom_vector(x_total[i])

    # Create test PHI matrix
    shape = (len(x_test), 5)
    phi_test = np.empty(shape)
    for i in range(len(x_test)):
        phi_test[i, :] = custom_vector(x_test[i])

    # Compute SVD
    U, S, Vh = np.linalg.svd(phi)

    # Invert Sigma
    sig = np.diag(S)
    filler = np.zeros([len(x_total) - len(S), len(S)])
    sig = np.vstack([sig, filler])

    # Compute Test Predictions
    temp0 = np.dot(sig.T, sig)
    temp1 = np.linalg.inv(temp0 + l_val * np.eye(len(temp0)))
    w = np.dot(Vh.T, np.dot(temp1, np.dot(sig.T, np.dot(U.T, y_total))))
    predictions = np.dot(phi_test, w)

    test_error = compute_rmse(y_test, predictions)

    plt.figure(1)
    plt.plot(x_test, y_test, '-b', label='Actual')
    plt.plot(x_test, predictions, '-r', label='Predicted')
    plt.title('Mauna Loa GLM Predictions at lambda=' + str(l_val))
    plt.xlabel('x_test')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig('mauna_loa_glm_estimates.png')

    return test_error


# ---------------------- Question 2 ----------------------

def glm_kernelized(x_train, x_valid, x_test, y_train, y_valid, y_test, l_val):

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    # Create and Populate the Gram Matrix (K)
    shape = (len(x_total), len(x_total))
    K = np.empty(shape)
    prev_computed = {}                          # Stores previously computed custom kernels
    for i in range(len(x_total)):
        for j in range(len(x_total)):
            a = x_total[i]
            b = x_total[j]
            # Add to previously computed dictionary
            if str((a, b)) not in prev_computed:
                prev_computed[str((a, b))] = custom_kernel(a, b)
                prev_computed[str((b, a))] = prev_computed[str((a, b))]
            # Add kernel to K (Gram) Matrix
            K[i, j] = prev_computed[str((a, b))]

    # Cholesky Factorization of K + lambda*1, here R is lower triangular
    R = np.linalg.cholesky((K + l_val*np.eye(len(K))))

    # Find inverse, P = inv(R) makes it quicker to find matrix inverse
    P = np.linalg.inv(R)

    # Estimate dual-variables alpha
    alp = np.dot(np.dot(P.T, P), y_total)

    # Compute predictions
    predictions = np.empty(np.shape(y_test))
    for i in range(len(x_test)):
        # Create k vector, containing the kernel products of x_test and all x_training points
        k = np.empty(np.shape(alp))
        for j in range(len(x_total)):
            k[j] = custom_kernel(x_test[i], x_total[j])
        # Make prediction for x_test[i]
        predictions[i] = np.dot(k.T, alp)

    # Compute model test_error
    test_error = compute_rmse(y_test, predictions)

    plt.figure(2)
    plt.plot(x_test, y_test, '-b', label='Actual')
    plt.plot(x_test, predictions, '-r', label='Predicted')
    plt.title('Mauna Loa Kernelized GLM Predictions at lambda=' + str(l_val))
    plt.xlabel('x_test')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.savefig('mauna_loa_glm_kernel_estimates.png')

    return test_error


def visualize_kernel():

    for i in range(2):
        y_vals = list()

        z = np.linspace(-0.1 + i, 0.1 + i, 100)
        z = np.array(z)

        for elem in z:
            y_vals.append(custom_kernel(i, elem))

        plt.figure(i + 3)
        plt.plot(z, y_vals, '-b', label='kernel')
        plt.title('kernel(' + str(i) + ', z+' + str(i) + ') over z')
        plt.xlabel('z')
        plt.ylabel('k')
        plt.legend(loc='lower right')
        plt.savefig('kernel' + str(i) + '.png')


# ---------------------- Question 3 ----------------------

def gaussian_rbf_glm_valid(x_train, x_valid, y_train, y_valid, dataset):

    theta_vals = [0.05, 0.1, 0.5, 1, 2]
    reg_vals = [0.001, 0.01, 0.1, 1]
    # Will store the validation_errors/validation_accuracy for each theta-regularization value pair
    results = {}

    # theta value only affects Gram Matrix and K Vector used in prediction
    for theta in theta_vals:

        # Create and Populate the Gram Matrix (K) at the current theta value
        shape = (len(x_train), len(x_train))
        K = np.empty(shape)                 # Gram Matrix
        prev_computed = {}                  # Store previously computed gaussian kernel
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                a = x_train[i]
                b = x_train[j]
                if str((a, b)) not in prev_computed:
                    prev_computed[str((a, b))] = gaussian_rbf(a, b, theta)
                    prev_computed[str((b, a))] = prev_computed[str((a, b))]
                # Add kernel to K (Gram) Matrix
                K[i, j] = prev_computed[str((a, b))]

        # Only alpha changes with regularization value, thus calculate k_vector (and extend to matrix form = kM)
        kM = np.empty((len(x_valid), len(x_train)))
        for i in range(len(x_valid)):
            # Create k vector, containing the kernel products of x_test and all x_training points
            k = list()
            vec = x_valid[i]
            for j in range(len(x_train)):
                k.append(gaussian_rbf(vec, x_train[j], theta))
            kM[i, :] = np.array(k)

        # K, kM matrix computed, compute test_errors for each regularization value at current theta
        for l_val in reg_vals:

            # Cholesky Factorization of K + lambda*1, here R is lower triangular
            R = np.linalg.cholesky((K + l_val * np.eye(len(K))))
            # Find inverse, P = inv(R) makes it quicker to find matrix inverse
            P = np.linalg.inv(R)
            # Estimate dual-variables alpha
            alp = np.dot(np.dot(P.T, P), y_train)

            # Compute Test RMSE for regresion datasets
            if dataset == 'mauna_loa' or dataset == 'rosenbrock':
                # Compute predictions
                predictions = np.dot(kM, alp)
                # Compute model validation error at current regularization-theta pair, and store in results
                results[(theta, l_val)] = compute_rmse(y_valid, predictions)

            # Compute Test Accuracy Ratio for classification datasets
            else:
                # Compute predictions
                predictions = np.argmax(np.dot(kM, alp), axis=1)
                y_valid0 = np.argmax(1 * y_valid, axis=1)
                # Compute model prediction accuracy at current regularization-theta pair, and store in results
                results[(theta, l_val)] = (predictions == y_valid0).sum() / len(y_valid0)

    # Acquire the optimal theta and regularization values, return them to be used for test set
    if dataset == 'mauna_loa' or dataset == 'rosenbrock':
        opt_res = np.inf
        for theta, l_val in results:
            if results[(theta, l_val)] < opt_res:
                opt_res = results[(theta, l_val)]
                opt_theta = theta
                opt_reg = l_val
    else:
        opt_res = np.NINF
        for theta, l_val in results:
            if results[(theta, l_val)] > opt_res:
                opt_res = results[(theta, l_val)]
                opt_theta = theta
                opt_reg = l_val

    return opt_theta, opt_reg, opt_res


def gaussian_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l_val, theta, dataset):

    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])

    # Create and Populate the Gram Matrix (K) at the current theta value
    shape = (len(x_total), len(x_total))
    K = np.empty(shape)  # Gram Matrix
    prev_computed = {}  # Store previously computed gaussian kernel
    for i in range(len(x_total)):
        for j in range(len(x_total)):
            a = x_total[i]
            b = x_total[j]
            if str((a, b)) not in prev_computed:
                prev_computed[str((a, b))] = gaussian_rbf(a, b, theta)
                prev_computed[str((b, a))] = prev_computed[str((a, b))]
            # Add kernel to K (Gram) Matrix
            K[i, j] = prev_computed[str((a, b))]

    # Only alpha changes with regularization value, thus calculate k_vector (and extend to matrix form = kM)
    kM = np.empty((len(x_test), len(x_total)))
    for i in range(len(x_test)):
        # Create k vector, containing the kernel products of x_test and all x_training points
        k = list()
        vec = x_test[i]
        for j in range(len(x_total)):
            k.append(gaussian_rbf(vec, x_total[j], theta))
        kM[i, :] = np.array(k)

    # K, kM matrix computed, compute test_error for each regularization value at current theta

    # Cholesky Factorization of K + lambda*1, here R is lower triangular
    R = np.linalg.cholesky((K + l_val * np.eye(len(K))))
    # Find inverse, P = inv(R) makes it quicker to find matrix inverse
    P = np.linalg.inv(R)
    # Estimate dual-variables alpha
    alp = np.dot(np.dot(P.T, P), y_total)

    if dataset == 'mauna_loa' or dataset == 'rosenbrock':
        # Compute predictions
        predictions = np.dot(kM, alp)
        # Compute model test_error at current regularization-theta pair
        test_error = compute_rmse(y_test, predictions)

    else:
        # Compute predictions
        predictions = np.argmax(np.dot(kM, alp), axis=1)
        y_test = np.argmax(1 * y_test, axis=1)
        # Compute model prediction accuracy at current regularization-theta pair
        test_error = (predictions == y_test).sum() / len(y_test)

    return test_error


def gaussian_rbf(x0, x1, theta):
    return math.exp(-((l2_norm(x0, x1))**2)/theta)


# ---------------------- Main Block ----------------------

if __name__ == '__main__':
    # All Dataset Names
    all_datasets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm', 'iris', 'mnist_small']
    regression_sets = ['mauna_loa', 'rosenbrock', 'pumadyn32nm']
    classification_sets = ['iris', 'mnist_small']

    # ---------------------- Question 1 ----------------------

    # print('------------------ Overall Results for Question 1 -------------------')
    # print('')
    #
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # l_val = glm_validation(x_train, x_valid, y_train, y_valid)
    # test_rmse = glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, l_val)
    #
    # print('Optimal Regularization Parameter (validation): ' + str(l_val))
    # print('Test Root Mean-Squared Error: ' + str(test_rmse))
    # print('')

    # ---------------------- Question 2 ----------------------

    # print('------------------ Overall Results for Question 2 -------------------')
    # print('')
    #
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # test_rmse = glm_kernelized(x_train, x_valid, x_test, y_train, y_valid, y_test, 14)
    #
    # visualize_kernel()
    #
    # print('Test Root Mean-Squared Error: ' + str(test_rmse))
    # print('')

    # ---------------------- Question 3 ----------------------

    # print('------------------ Overall Results for Question 3 -------------------')
    # print('')
    #
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # theta, reg, valid_error = gaussian_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'mauna_loa')
    # test_rmse = gaussian_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'mauna_loa')
    # print('--- Results for mauna_loa ---')
    # print('Optimal Lengthscale: ' + str(theta))
    # print('Optimal Regularizer: ' + str(reg))
    # print('Valid RMSE: ' + str(valid_error))
    # print('Test RMSE: ' + str(test_rmse))
    # print('')

    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
    # theta, reg, valid_error = gaussian_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'rosenbrock')
    # test_rmse = gaussian_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'rosenbrock')
    # print('--- Results for rosenbrock ---')
    # print('Optimal Lengthscale: ' + str(theta))
    # print('Optimal Regularizer: ' + str(reg))
    # print('Valid RMSE: ' + str(valid_error))
    # print('Test RMSE: ' + str(test_rmse))
    # print('')

    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    # theta, reg, valid_ratio = gaussian_rbf_glm_valid(x_train, x_valid, y_train, y_valid, 'iris')
    # test_ratio = gaussian_rbf_glm_test(x_train, x_valid, x_test, y_train, y_valid, y_test, reg, theta, 'iris')
    # print('--- Results for iris ---')
    # print('Optimal Lengthscale: ' + str(theta))
    # print('Optimal Regularizer: ' + str(reg))
    # print('Valid Accuracy Ratio: ' + str(valid_ratio))
    # print('Test Accuracy Ratio: ' + str(test_ratio))
    #
    # print('')

    # ---------------------- Question 4 ----------------------
    # Done on paper
    # ---------------------- Question 5 ----------------------
    # Done on paper
