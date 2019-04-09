import numpy as np
import math 
import scipy
from matplotlib import pyplot as plt
from data_utils import load_dataset


__author__ = 'Christopher Agia (1003243509)'
__date__ = 'April 2, 2019'


# Useful Utility Functions
def compute_accuracy_ratio(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)


def X_mat(x_data):
    X = np.ones((len(x_data), len(x_data[0]) + 1))
    X[:, 1:] = x_data
    return X


def sigmoid(z):
    return np.divide(1, np.add(1, np.exp(-1*z)))


def log_likelihood(x_prod, y_act):
    log_p = np.dot(y_act.T, np.log(sigmoid(x_prod))) + np.dot(np.subtract(1, y_act).T, np.log(np.subtract(1, sigmoid(x_prod))))
    return log_p


def likelihood_grad(X, x_prod, y_act):
    grad = np.zeros(np.shape(X[0]))
    for i in range(len(x_prod)):
        grad += (y_act[i] - sigmoid(x_prod[i])) * X[i]
    return grad


def likelihood_2grad(X, x_prod):
    hess = np.zeros((len(X[0]), len(X[0])))
    sig_vec = np.multiply(sigmoid(x_prod), sigmoid(x_prod) - 1)
    for i in range(len(x_prod)):
        hess = np.add(hess, sig_vec[i] * np.outer(X[i], X[i].T))
    return hess


def log_prior(w, sigma):
    return -len(w)/2 * np.log(2 * np.pi) - len(w)/2 * np.log(sigma) - 1/(2 * sigma) * np.dot(w.T, w)


def prior_grad(w, sigma):
    return -1/sigma * w


def prior_2grad(w, sigma):
    return -1/sigma * np.eye(len(w))


def log_g(hessian):
    return 1/2 * np.log(np.linalg.det(-1 * hessian)) - len(hessian) / 2 * np.log(2 * np.pi)


def likelihood(x, y):
    likelihood = 1
    for i in range(len(x)):
        likelihood *= (sigmoid(x[i]) ** y[i]) * ((1 - sigmoid(x[i])) ** (1 - y[i]))
    return likelihood


def prior_like(w, variance):
    prior = 1
    for i in range(len(w)):
        prior *= 1 / math.sqrt(2 * math.pi * variance) * math.exp(-(w[i] ** 2) / (2 * variance))
    return prior


def proposal_like(w, proposal_var, mean):
    proposal = 1
    for i in range(len(w)):
        proposal *= 1 / math.sqrt(2 * math.pi * proposal_var) * math.exp(-((mean[i] - w[i]) ** 2) / (2 * proposal_var))
    return proposal


def r(x, y, w, prior_var, proposal_var, mean):
    return likelihood(x, y) * prior_like(w, prior_var) / proposal_like(w, proposal_var, mean)


def proposal(mean, variance):
    return np.random.multivariate_normal(mean=mean, cov=np.eye(np.shape(mean)[0]) * variance)


def sample_weights(sample_size, mean, variance):
    w = list()
    for i in range(sample_size):
        w.append(proposal(mean, variance))
    return w


def compute_log_likelihood(y_pred, y):
    log_p = np.dot(y.T, np.log(y_pred)) + np.dot(np.subtract(1, y).T, np.log(np.subtract(1, y_pred)))
    return log_p

# --- Question 1 ---

def laplace_approx(x_train, x_test, y_train, y_test, rate):

    # variances
    sigma_var = [0.5, 1, 2]

    # acquire x matrices
    X_train = X_mat(x_train)
    X_test = X_mat(x_test)

    # marginal likelihoods
    marg_likelihoods = dict()

    for variance in sigma_var:

        # initialize weights and iteration count
        w = np.zeros(np.shape(X_train[0]))
        num_it = 0

        # initialize first gradient

        x_prod = np.reshape(np.dot(X_train, w), np.shape(y_train))
        post_grad = likelihood_grad(X_train, x_prod, y_train) + prior_grad(w, variance)

        # breakout when gradients become near zero
        while max(post_grad) > 10**(-9):

            # compute estimates, gradients, and update
            x_prod = np.dot(X_train, w)
            post_grad = likelihood_grad(X_train, x_prod, y_train) + prior_grad(w, variance)
            w = np.add(w, rate*post_grad)

            num_it += 1

        # compute hessian at MAP solution
        hessian = likelihood_2grad(X_train, x_prod) + prior_2grad(w, variance)

        # compute marginal likelihood
        marg_likelihoods[variance] = log_likelihood(x_prod, y_train) + log_prior(w, variance) - log_g(hessian)

        print('Number iterations for variance = ' + str(variance) + ' is ' + str(num_it))

    return marg_likelihoods


# --- Question 2 ---

def importance_sampling(x_train, x_valid, x_test, y_train, y_valid, y_test, mean, sample_range=[5, 10, 20, 50, 100, 500], visual=False):

    # prior variance
    prior_variance = 1

    # proposal distribution metrics
    variances = [1, 2, 5, 10]

    # reconstruct y sets
    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)

    # make X matrices
    X_train = X_mat(x_train)
    X_valid = X_mat(x_valid)
    X_test = X_mat(x_test)

    min_ll = np.inf
    for sample_size in sample_range:
        for proposal_variance in variances:

            # store all test set classifications
            valid_pred = np.zeros(np.shape(y_valid))
            valid_discrete_pred = np.zeros(np.shape(y_valid))

            # sample s number of weights
            w = sample_weights(sample_size, mean, proposal_variance)

            # compute predictions over the whole validation set
            for d in range(len(X_valid)):

                # inner sample over j
                r_sum = 0
                for j in range(sample_size):
                    # compute and sum r(w_js)
                    r_sum += r(np.dot(X_train, w[j]), y_train, w[j], prior_variance, proposal_variance, mean)

                # outer sample over i
                pred_sum = 0
                for i in range(sample_size):
                    # compute sigmoid prediction of test point
                    y_star = sigmoid(np.dot(X_valid[d], w[i]))
                    # add to prediction summation 
                    pred_sum += y_star*r(np.dot(X_train, w[i]), y_train, w[i], prior_variance, proposal_variance, mean)/r_sum
                
                # make classification (discretized and continuous)
                valid_pred[d] = pred_sum
                if pred_sum > 0.5:
                    valid_discrete_pred[d] = 1
                elif pred_sum < 0.5:
                    valid_discrete_pred[d] = 0
                else:
                    valid_discrete_pred[d] = -1

            valid_log_likelihood = -compute_log_likelihood(valid_pred, y_valid)
            if valid_log_likelihood < min_ll:
                min_ll = valid_log_likelihood
                min_acc = compute_accuracy_ratio(valid_discrete_pred, y_valid)
                opt_var = proposal_variance
                opt_ss = sample_size

    # re-make X matrices with combined training and validation set
    x_train = np.vstack((x_train, x_valid))
    X_train = X_mat(x_train)
    y_train = np.vstack((y_train, y_valid))

    # store all test set classifications
    test_pred = np.zeros(np.shape(y_test))
    test_discrete_pred = np.zeros(np.shape(y_test))

    # sample s number of weights
    w = sample_weights(opt_ss, mean, opt_var)

    # compute predictions over the whole test set
    for d in range(len(X_test)):

        # inner sample over j
        r_sum = 0
        for j in range(opt_ss):
            # compute and sum r(w_js)
            r_sum += r(np.dot(X_train, w[j]), y_train, w[j], prior_variance, opt_var, mean)

        # outer sample over i
        pred_sum = 0
        for i in range(opt_ss):
            # compute sigmoid prediction of test point
            y_star = sigmoid(np.dot(X_test[d], w[i]))
            # add to prediction summation 
            pred_sum += y_star*r(np.dot(X_train, w[i]), y_train, w[i], prior_variance, opt_var, mean)/r_sum

        # make classification (discretized and continuous)
        prediction = pred_sum
        test_pred[d] = prediction
        if prediction > 0.5:
            test_discrete_pred[d] = 1
        elif prediction < 0.5:
            test_discrete_pred[d] = 0
        else:
            test_discrete_pred[d] = -1

    test_accuracy_ratio = compute_accuracy_ratio(test_discrete_pred, y_test)
    test_log_likelihood = compute_log_likelihood(test_pred, y_test)

    if visual:
        # sample s number of weights
        w = sample_weights(5000, mean, 2)

        # inner sample over j
        r_sum = 0
        for j in range(5000):
            # compute and sum r(w_js)
            r_sum += r(np.dot(X_train, w[j]), y_train, w[j], prior_variance, 2, mean)

        # outer sample over i
        posterior = list()
        for i in range(5000):
            # add to prediction summation
            posterior.append(r(np.dot(X_train, w[i]), y_train, w[i], prior_variance, 2, mean) / r_sum)

        visualize_posterior(mean, 2, posterior, w)

    return -test_log_likelihood, test_accuracy_ratio, opt_var, opt_ss, min_ll, min_acc


# --- Question 3 ---

def generate_hastings_sample(X, y, sample_size, variance, map):
    sample_means = list()
    samples = list()

    # must burned in 10000 iterations before storing sample
    w_mean = map
    w_i = proposal(w_mean, variance)
    burned_in = False

    while len(samples) < sample_size:

        if not burned_in:
            # burn in 10000 iterations
            for j in range(1000):
                # generate uniform random variable, and random sample
                u = np.random.uniform()
                w_star = proposal(w_i, variance)

                if u < min(1, likelihood(np.dot(X, w_star), y) * prior_like(w_star, 1) / likelihood(np.dot(X, w_i),
                                                                                             y) / prior_like(w_i, 1)):
                    w_mean = w_i
                    w_i = w_star

            samples.append(w_i)
            sample_means.append(w_mean)
            burned_in = True
            continue

        for j in range(100):

            # generate uniform random variable, and random sample
            u = np.random.uniform()
            w_star = proposal(w_i, variance)

            if u < min(1, likelihood(np.dot(X, w_star), y) * prior_like(w_star, 1) / likelihood(np.dot(X, w_i),
                                                                                             y) / prior_like(w_i, 1)):
                w_mean = w_i
                w_i = w_star

        # collect sample every 100 iterations for thinning process (after 10000 burn in)
        sample_means.append(w_mean)
        samples.append(w_i)

    r_sum = 0
    for i in range(sample_size):
        # compute and sum r(w_js)
        r_sum += r(np.dot(X, samples[i]), y, samples[i], 1, variance, sample_means[i])

    return samples, sample_means, r_sum


def metropolis_hastings(x_train, x_valid, x_test, y_train, y_valid, y_test, map, visual=False):
    # prior variance
    prior_variance = 1

    # sample size
    sample_size = 100

    # proposal distribution metrics
    variances = [1, 2, 5, 10, 15]

    # reconstruct y sets
    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)

    # make X matrices
    X_train = X_mat(x_train)
    X_valid = X_mat(x_valid)
    X_test = X_mat(x_test)

    min_ll = np.inf
    for proposal_variance in variances:

        # store all test set classifications
        valid_pred = np.zeros(np.shape(y_valid))
        valid_discrete_pred = np.zeros(np.shape(y_valid))

        # sample s number of weights, and obtain r_sum
        w, means, r_sum = generate_hastings_sample(X_train, y_train, sample_size, proposal_variance, map)

        # compute predictions over the whole validation set
        for d in range(len(X_valid)):

            # outer sample over i
            pred_sum = 0
            for i in range(sample_size):
                # compute sigmoid prediction of test point
                y_star = sigmoid(np.dot(X_valid[d], w[i]))
                # add to prediction summation
                pred_sum += y_star * r(np.dot(X_train, w[i]), y_train, w[i], prior_variance, proposal_variance,
                                       means[i]) / r_sum

            # make classification (discretized and continuous)
            valid_pred[d] = pred_sum
            if pred_sum > 0.5:
                valid_discrete_pred[d] = 1
            elif pred_sum < 0.5:
                valid_discrete_pred[d] = 0
            else:
                valid_discrete_pred[d] = -1

        valid_log_likelihood = -compute_log_likelihood(valid_pred, y_valid)
        if valid_log_likelihood < min_ll:
            min_ll = valid_log_likelihood
            min_acc = compute_accuracy_ratio(valid_discrete_pred, y_valid)
            opt_var = proposal_variance

    # re-make X matrices with combined training and validation set
    x_train = np.vstack((x_train, x_valid))
    X_train = X_mat(x_train)
    y_train = np.vstack((y_train, y_valid))

    # store all test set classifications
    test_pred = np.zeros(np.shape(y_test))
    test_discrete_pred = np.zeros(np.shape(y_test))

    # sample s number of weights, and obtain r_sum
    w, means, r_sum = generate_hastings_sample(X_train, y_train, sample_size, opt_var, map)

    # store predictive posterior results for test point 9 and 10
    ninth_pred = list()
    tenth_pred = list()

    # compute predictions over the whole test set
    for d in range(len(X_test)):

        # outer sample over i
        pred_sum = 0
        for i in range(sample_size):
            # compute sigmoid prediction of test point
            y_star = sigmoid(np.dot(X_test[d], w[i]))
            if d == 9:
                ninth_pred.append(y_star)
            if d == 10:
                tenth_pred.append(y_star)
            # add to prediction summation
            pred_sum += y_star * r(np.dot(X_train, w[i]), y_train, w[i], prior_variance, opt_var, means[i]) / r_sum

        # make classification (discretized and continuous)
        prediction = pred_sum
        test_pred[d] = prediction
        if prediction > 0.5:
            test_discrete_pred[d] = 1
        elif prediction < 0.5:
            test_discrete_pred[d] = 0
        else:
            test_discrete_pred[d] = -1

    test_accuracy_ratio = compute_accuracy_ratio(test_discrete_pred, y_test)
    test_log_likelihood = compute_log_likelihood(test_pred, y_test)

    if visual:
        visualize_predictive_posterior([ninth_pred, tenth_pred])

    return -test_log_likelihood, test_accuracy_ratio, opt_var, min_ll, min_acc


# --- Visualization Functions ---

def visualize_posterior(mean, variance, posterior, w):
    # five components to each weight vector
    for i in range(5):
        weights = list()
        # extract specific component of all weights
        for j in range(len(w)):
            weights.append(w[j][i])

        # zip weights and posterior
        weights, posterior = zip(*sorted(zip(weights, posterior)))

        # set up gaussian
        z = np.polyfit(weights, posterior, 1)
        z = np.squeeze(z)
        p = np.poly1d(z)

        # plot
        w_all = np.arange(min(weights), max(weights), 0.001)
        q_w = scipy.stats.norm.pdf(w_all, mean[i], variance)
        plt.figure(i)
        plt.title("Posterior visualization: q(w) mean=" + str(round(mean[i], 2)) + " var=" + str(variance))
        plt.xlabel("w[" + str(i) + "]")
        plt.ylabel("Probability")
        plt.plot(w_all, q_w, '-b', label="Proposal q(w)")
        plt.plot(weights, posterior, 'or', label="Posterior P(w|X,y)")
        plt.plot(weights, p(weights),"r--")
        plt.legend(loc='upper right')
        plt.savefig("weight_vis_" + str(i) + ".png")


def visualize_predictive_posterior(predictive_posterior):
    for i in range(len(predictive_posterior)):
        plt.figure(i)
        plt.title('Predictive Posterior for Flower ' + str(i+9))
        plt.xlabel('Pr(y*|x*, w(i))')
        plt.xlim((0, 1))
        plt.ylabel('# Occurrences')
        plt.hist(predictive_posterior[i], bins=20)
        plt.savefig('flower_' + str(i+9) + '.png')


# --- Main Block ---
if __name__ == '__main__':
    print('')
    # Set True for Question 1
    Q1 = False
    # Set True for Question 2
    Q2 = False
    Q2_vis = False
    # Set True for Question 3
    Q3 = False
    Q3_vis = False

    # Import Dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    # --- Question 1 ---
    if Q1:
        print('--- Results for Question 1 ---')
        print('')
        x_train, x_test = np.vstack((x_train, x_valid)), x_test
        y_train, y_test = np.vstack((y_train, y_valid)), y_test
        marginal_likelihood = laplace_approx(x_train, x_test, y_train, y_test, 0.001)
        for var in marginal_likelihood:
            print('For variance = ' + str(var) + ' , with a marginal likelihood of ' + str(marginal_likelihood[var]))
        print('')

    # --- Question 2 ---
    if Q2:
        print('--- Results for Question 2 ---')
        print('')
        map_mean = [-0.87805271, 0.29302957, -1.2347739, 0.67815586, -0.89401743]
        if Q2_vis:
            test_ll, test_ar, prop_var, ss, valid_ll, valid_ar = importance_sampling(x_train, x_valid, x_test, y_train,
                                                                                     y_valid,
                                                                                     y_test, map_mean, visual=True)
        else:
            test_ll, test_ar, prop_var, ss, valid_ll, valid_ar = importance_sampling(x_train, x_valid, x_test, y_train,
                                                                                     y_valid, y_test, map_mean)
        print('Proposal Distribution Variance: ' + str(prop_var))
        print('Sample Size: ' + str(ss))
        print('Validation Log-likelihood: ' + str(valid_ll))
        print('Validation Accuracy Ratio: ' + str(valid_ar))
        print('Test Log-likelihood: ' + str(test_ll))
        print('Test Accuracy Ratio: ' + str(test_ar))
        print('')
        if Q2_vis:
            print('See current directory for posterior visualization')
        print('')

    # --- Question 3 ---
    if Q3:
        print('--- Results for Question 3 ---')
        print('')
        map_mean = [-0.87805271, 0.29302957, -1.2347739, 0.67815586, -0.89401743]
        if Q3_vis:
            test_ll, test_ar, prop_var, valid_ll, valid_ar = metropolis_hastings(x_train, x_valid, x_test, y_train,
                                                                                 y_valid, y_test, map_mean, visual=True)
        else:
            test_ll, test_ar, prop_var, valid_ll, valid_ar = metropolis_hastings(x_train, x_valid, x_test, y_train,
                                                                                 y_valid, y_test, map_mean)
        print('Proposal Distribution Variance: ' + str(prop_var))
        print('Sample Size: ' + str(100))
        print('Validation Log-likelihood: ' + str(valid_ll))
        print('Validation Accuracy Ratio: ' + str(valid_ar))
        print('Test Log-likelihood: ' + str(test_ll))
        print('Test Accuracy Ratio: ' + str(test_ar))
        print('')
        if Q3_vis:
            print('See current directory for flowers 9 and 10 predictive posterior histograms')
        print('')
