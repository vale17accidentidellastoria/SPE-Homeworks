import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import statsmodels.api as sm

COLORS = {'blue':'#0a0add', 'red':'#FF0000', 'black':'#000000'} 

def loadCSV(csv_file): #TODO: Remember to modify it according to data of exercise 2
    dataset = pd.read_csv(csv_file, header=None) #here we are working with Pandas DataFrame
    #print(dataset.shape)

    data_points = []
    column_dim = dataset.shape[1]  # save the number of columns
    row_dim = dataset.shape[0]  # save the number of columns
    if (column_dim == 1):  # check if we have data organized in a single column
        data_points = dataset[0].values.tolist()  # convert the values of the column into a list of values
    elif (column_dim > 1):  # check if data is expressed as a matrix of values (multiple columns)
        data_points = dataset.values.tolist()
        if (row_dim == 1):  # check if data is expressed as a single row
            data_points = data_points[0]
    return data_points


def computeBins(data):
    """ Computes the necessary number of bins for the histogram according to the input data """
    num_observation = len(data)  # the number of observations
    data_range = max(data) - min(data)  # range is the difference between minimum value and maximum value
    num_intervals = int(round(math.sqrt(num_observation)))
    binwidth = data_range / num_intervals
    return np.arange(min(data), max(data) + binwidth, binwidth)


def plotHistogram(data, d=False):
    bins_hist = computeBins(data)
    plt.hist(data, bins=bins_hist, density=d)
    plt.show()


def plotPDFsOnHistogram(data, mu_gauss, sigma_gauss):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    bins_hist = computeBins(data)
    ax1.hist(data, bins=bins_hist, density=True, alpha=0.8, color='g')

    ax2 = ax1.twinx()

    plt.yticks([])

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    g1_pdf = st.norm.pdf(x, mu_gauss[0], sigma_gauss[0])
    g2_pdf = st.norm.pdf(x, mu_gauss[1], sigma_gauss[1])
    g3_pdf = st.norm.pdf(x, mu_gauss[2], sigma_gauss[2])

    ax2.plot(x, g1_pdf, 'k', linewidth=2, color="r")
    ax2.plot(x, g2_pdf, 'k', linewidth=2, color="y")
    ax2.plot(x, g3_pdf, 'k', linewidth=2, color="b")

    align_yaxis(ax1, 0, ax2, 0)

    plt.show()


def plotBarDiagram(items, probs):
    plt.bar(items, probs, width=0.2)
    plt.show()


def plotHistogramExp(exp_rv):
    mybins = computeBins(exp_rv)
    plt.hist(exp_rv, bins=mybins, density=True)
    P = st.expon.fit(exp_rv)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    rP = st.expon.pdf(x, *P)

    plt.plot(x, rP)
    plt.show()


def plotQQ(rvs, lamb, N, dist="expon"):
    if dist == "expon":
        # fig = sm.qqplot(np.array(exp_rv), dist=st.expon, fit=True, line='45')
        fig = sm.qqplot(np.array(rvs), dist=st.expon, scale=(1 / lamb), line='45')  # the scale for an exponential distribution
    plt.title("QQ Plot (" + str(N) + " draws) against exponential quantiles with average value = " + str(lamb))
    plt.show()

def plotScatter(x, y, title='Scatter Plot of data', x_label='x', y_label='y', show=True, color=COLORS['blue']):
    area = np.pi
    plt.scatter(x, y, s=area, c=color, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show:
        plt.show()


def polynomialMatrix(data, poly_grade=6):
    poly_matrix = np.ones(shape=(data.size, poly_grade), dtype='float64')
    for i in range(data.size):
        for j in range(poly_grade):
            poly_matrix[i][j] = data[i]**(j)
    return poly_matrix

def computeLeastSquares(A, y):
    b = np.ones(shape=A.shape[1])
    b = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A), y))
    print(b)
    return b

def applyPolyOutput(data, polynomial_vector):
    for i in range(data.size):
        tmp_data = 0
        for j in range(polynomial_vector.size):
            tmp_data += (data[i]**j)*polynomial_vector[j]
        data[i] = tmp_data
    return data

def exercise1(data, poly_grade=[4, 6]):
    data_matrix = np.around(np.array(data), 3)
    plotScatter(data_matrix[:,0], data_matrix[:,1], title='Scatter Plot of Exercise1', x_label='Time of measurement', y_label='output', show=False)
    exercise1LS(data, data_matrix, poly_grade=poly_grade[0], color=COLORS['black'])

    data_matrix = np.around(np.array(data), 3)
    data_y = exercise1LS(data, data_matrix, poly_grade=poly_grade[1], color=COLORS['red'], show_plot=True)

    data_matrix = np.around(np.array(data), 3)
    detrend_y = data_matrix[:,1]-data_y
    plotScatter(data_matrix[:,0], detrend_y, title='Scatter Plot of Exercise1', x_label='Time of measurement', y_label='output', show=True)

    mean,std=st.norm.fit(detrend_y)

    plt.hist(detrend_y, bins=12, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = st.norm.pdf(x, mean, std)
    plt.plot(x, y)
    plt.show()

    print('The mean of the distribution is {}, the variance is {} and the prediction interval at level 95% is [{}, {}].'.format(np.around(mean, 3), np.around(std**2, 3), mean-(1.96*std), mean+(1.96*std)))

    st.probplot(detrend_y, dist="norm", plot=plt)
    plt.show()


def exercise1LS(data, data_matrix, color, poly_grade=6, show_plot=False):
    matrix_A = polynomialMatrix(data_matrix[:,0], poly_grade=poly_grade)
    matrix_A = np.around(matrix_A, 3)

    vector_b = computeLeastSquares(matrix_A, data_matrix[:,1])
    data_y_ls = applyPolyOutput(data_matrix[:,0], vector_b)

    data_matrix = np.around(np.array(data), 3)
    plotScatter(data_matrix[:,0], data_y_ls, title='Scatter Plot of Exercise1', x_label='Time of measurement', y_label='output', color=color, show=show_plot)
    return data_y_ls

def bernoulliRNG(s_prob=0.5):
    return 1 if random(0, 1, 0.00001) <= s_prob else 0

def countOnes(elements, s_prob=0.05):
    counter = 0
    for e in elements:
        if e <= s_prob:
            counter += 1
    return counter  

def exercise3(n_trials=10000, n_exp=100, p_success=0.05):
    trials = []
    trials_success = []
    for t in range(n_trials):
        exp = np.random.uniform(0, 1, n_exp)
        p_trial = countOnes(exp, p_success) / float(n_exp)
        trials.append(p_trial)
        trials_success.append(countOnes(exp, p_success))
        
    #print(trials)

    val, cnt = np.unique(trials_success, return_counts=True)
    prop = cnt / n_trials
    print('val', val)

    plt.subplot(1, 3, 1)
    plt.title('Empirical PMF')
    plt.bar(val, prop)
    plt.ylabel("Probability")
    plt.xlabel("1s in the trial")

    x = val
    binomial = st.binom.pmf(x, n_exp, p_success) #(np.arange(0,21), n_trials, p_success) #.pmf(n_trials*p_success, n_trials, p_success)
    print(binomial)

    plt.subplot(1, 3, 2)
    plt.title('Theoretical Binomial PMF')
    plt.bar(x, binomial)
    plt.ylabel("Probability")
    plt.xlabel("1s in the trial")

    poisson = st.poisson.pmf(x, n_exp*p_success) #(np.arange(0,21), n_trials, p_success) #.pmf(n_trials*p_success, n_trials, p_success)
    print(poisson)

    plt.subplot(1, 3, 3)
    plt.title('Theoretical Poisson PMF')
    plt.bar(x, poisson)
    plt.ylabel("Probability")
    plt.xlabel("1s in the trial")

    plt.show()

def intervalError(n_s, n_f, n, z=1.96): 
    return (1.96/n)*math.sqrt(n_s*(1-(n_s/float(n))))

def countSuccess(data):
    counter = 0
    for e in data:
        if e == 1:
            counter += 1
    return counter/float(len(data))

def monteCarloSim(n_s, n):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    d = np.around(math.sqrt(x**2+y**2), 3)

    #print('Point ({}, {}) with distance from origin of {}.'.format(x, y, d))

    success = False
    if d <= 1.0:
        n_s += 1
        success = True
    n += 1
    return success, n_s, n, 4*(n_s/float(n))

def meanSuccessProb(n_s_p):
    sum = 0
    for p in n_s_p:
        sum += p
    return sum/float(len(n_s_p))


def exercise5(z=1.96):
    n_s = 0
    n = 0
    n_s_p = []
    pi_s = []
    ie = 0
    while ie >= 0.01 or n_s < 6 or n-n_s < 6:
        success, n_s, n, pi = monteCarloSim(n_s, n)
        n_s_p.append(n_s/float(n))
        pi_s.append(pi)
        mean = meanSuccessProb(n_s_p)
        ie = intervalError(n_s, n-n_s, n)
        #print('n: {} | n_s: {} | Mean: {} | Estimated pi: {}... | IntervalError: {}, condition: {}'.format(n, n_s, mean, pi, ie, ie <= 0.01))
    
    mean = meanSuccessProb(n_s_p)
    ie = intervalError(n_s, n-n_s, n)
    print('n: {} | n_s: {} | Estimated pi: {}... | CI is [{}, {}]'.format(n, n_s, pi_s[len(pi_s)-1], np.around(mean-ie, 3), np.around(mean+ie, 3)))


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def EMalgorithm(data, n, num_iterations, update_prior_belief=False):
    d_length = len(data)
    p_gauss, mu_gauss, sigma_gauss = setPriorBelief()
    for i in range(0, num_iterations):
        if (i >= 1) and update_prior_belief:
            p_gauss = updatePriorBelief(gauss, d_length, n)
        p_x_gauss = runExpectationStep(data, n, mu_gauss, sigma_gauss)
        gauss = setPosteriorBelief(d_length, p_gauss, p_x_gauss)
        mu_gauss, sigma_gauss = runMaximizationStep(data, n, gauss, mu_gauss, sigma_gauss)

    return mu_gauss, sigma_gauss


def setPriorBelief():
    m = np.array([1 / 3, 1 / 3, 1 / 3])  # We deal with three distributions
    p_gauss = m / np.sum(m)
    mu_gauss = [5, -4, 2]
    sigma_gauss = [4, 3, 6]
    return p_gauss, mu_gauss, sigma_gauss


def updatePriorBelief(gauss, length, n):
    p_g = []
    for i in range(0, n):
        p_g.append(np.sum(gauss[i]) / length)
    return p_g


def runExpectationStep(data, n, mu_gauss, sigma_gauss):
    p_x_gauss = []
    for i in range(0, n):
        p_x_gauss.append(st.norm.pdf(data, loc=mu_gauss[i], scale=sigma_gauss[i]))
    return p_x_gauss


def setPosteriorBelief(len_data, p_gauss, p_x_gauss):
    gauss1 = []
    gauss2 = []
    gauss3 = []

    for i in range(0, len_data):
        den = (p_x_gauss[0][i] * p_gauss[0] + p_x_gauss[1][i] * p_gauss[1] + p_x_gauss[2][i] * p_gauss[2])
        gauss1.append((p_x_gauss[0][i] * p_gauss[0]) / den)
        gauss2.append((p_x_gauss[1][i] * p_gauss[1]) / den)
        gauss3.append((p_x_gauss[2][i] * p_gauss[2]) / den)

    return [gauss1, gauss2, gauss3]


def runMaximizationStep(data, n, gauss, mu_gauss, sigma_gauss):
    for i in range(0, n):
        mu_gauss[i] = np.sum(np.multiply(gauss[i], data)) / np.sum(gauss[i])
        diff = [x - mu_gauss[i] for x in data]
        sigma_gauss[i] = math.sqrt(np.sum(np.multiply(gauss[i], np.power(diff, 2))) / np.sum(gauss[i]))

    return mu_gauss, sigma_gauss

def extractItemsProbs(data):
    items = []
    num_occurrences = []
    probs = []
    for i in range(0, len(data)):
        items.append(data[i][0])
    for i in range(0, len(data)):
        num_occurrences.append(data[i][1])
    for i in range(0, len(num_occurrences)):
        probs.append(num_occurrences[i] / np.sum(num_occurrences))
    return items, probs, num_occurrences


def setProbTriangularDistr(items):
    probs = []
    for k in range(0, len(items) + 2):
        if 2 <= k <= 7:
            probs.append((k - 1) / 36)
        elif 8 <= k <= 12:
            probs.append((13 - k) / 36)
    return probs


def runChiSquaredTest(samples, pr):
    sample_size_n = np.sum(samples)
    test_statistic = 0
    alpha = 0.05

    for i in range(0, len(samples)):
        test_statistic += ((pow((samples[i] - sample_size_n * probs_2[i]), 2)) / (sample_size_n * pr[i]))
    degrees_of_freedom = len(samples) - 1
    p_value = 1 - st.chi2.cdf(test_statistic, degrees_of_freedom)

    if p_value < alpha:
        chi_res = False
    else:
        chi_res = True

    return test_statistic, p_value, chi_res, alpha


def generateExpRV(exp_lambda):
    U = random.uniform(0, 1)
    x = - (math.log(U) / exp_lambda)
    return x


def runCDFInversion(N, avg_lambda):
    rvs_exp = []
    for i in range(0, N):
        rvs_exp.append(generateExpRV(avg_lambda))
    return rvs_exp


# --------------------------------------------

#Exercise 1
data1 = loadCSV("data_hw2/data_ex1.csv")

exercise1(data1, poly_grade=[2, 5])

print("\n####################")

# --------------------------------------------

# Exercise 2
data2 = loadCSV("data_hw2/data_ex2.csv")
NUM_GAUSS = 3
NUM_ITERATIONS = 50

plotHistogram(data2, d=False)

print("\nExercise 2")

print("\t\tEM running in", NUM_ITERATIONS, "iterations!")

mu_gauss, sigma_gauss = EMalgorithm(data2, NUM_GAUSS, NUM_ITERATIONS, update_prior_belief=False)
plotPDFsOnHistogram(data2, mu_gauss, sigma_gauss)
print("\t 1. The means for the 3 Gaussians distributions are:", mu_gauss)
print("\t\tThe standard deviations for the 3 Gaussians distributions are:", sigma_gauss)

print("\t\t----------------------")

mu_gauss_up_prior, sigma_gauss_up_prior = EMalgorithm(data2, NUM_GAUSS, NUM_ITERATIONS, update_prior_belief=True)
plotPDFsOnHistogram(data2, mu_gauss_up_prior, sigma_gauss_up_prior)
print("\t\tThe means for the 3 Gaussians distributions (with prior update) are:", mu_gauss_up_prior)
print("\t\tThe standard deviations for the 3 Gaussians distributions (with prior update) are:", sigma_gauss_up_prior)

print("\t 2. Histograms with PDFs of Gaussians plotted!")


print("\n####################")

# --------------------------------------------

#Exercise 3

exercise3()
exercise3(n_exp=100, p_success=0.9)
#print('results {}.'.format(results))
#print('Success probability is {}.'.format(sp/val))

print("\n####################")

# --------------------------------------------

# Exercise 4
data4 = [[2, 1], [3, 4], [4, 2], [5, 7], [6, 10], [7, 9], [8, 9], [9, 14], [10, 7], [11, 5], [12, 3]]
items, probs_1, num_occurrences = extractItemsProbs(data4)

print("\nExercise 4")

print("\t 1. The PMF of the distribution is")
for i in range(0, len(items)):
    print("\t\tP[ X =", items[i], "] =", probs_1[i])

plotBarDiagram(items, probs_1)

probs_2 = setProbTriangularDistr(items)

T, p_value, chi_result, alpha = runChiSquaredTest(num_occurrences, probs_2)
print("\t 2. The test statistic T is", T)
print("\t\tThe pvalue is", p_value)

if chi_result:
    print("\t\tpvalue", p_value, ">", alpha, ": The null hypothesis should NOT be rejected!")
else:
    print("\t\tpvalue", p_value, "<", alpha, ": The null hypothesis should be rejected!")

plotBarDiagram(items, probs_2)

print("\n####################")

# --------------------------------------------

#Exercise 5
exercise5()

print("\n####################")

# --------------------------------------------

# Exercise 6

N_tr = 1000
avg_lambda1 = 2
exp_rv = runCDFInversion(N_tr, avg_lambda1)

print("\nExercise 6")

plotHistogramExp(exp_rv)

print("\t 1. Histogram that shows generation of exponential variates plotted!")
print("\t\tThe N_tr={}".format(N_tr),
      "exponential random variates with average value equal to 2 generated through CDF inversion method are:")
print("\t\t{}".format(exp_rv))

print("\t 2. Plotting QQ-Plot against quantiles of exponential distribution with average value =", avg_lambda1)
plotQQ(exp_rv, avg_lambda1, N_tr, dist="expon")

#Drawing qqplot against the quantiles of an exponential distribution with different average value
avg_lambda2 = 4

print("\t 3. Plotting QQ-Plot against quantiles of exponential distribution with average value =", avg_lambda2)
plotQQ(exp_rv, avg_lambda2, N_tr, dist="expon")

N_tr2 = 2*N_tr
exp_rv2 = runCDFInversion(N_tr2, avg_lambda1)

print("\t\tPlotting QQ-Plot against quantiles of exponential distribution with average value =", avg_lambda1, "and increasing by 2x the number of exponential draws")
plotQQ(exp_rv2, avg_lambda1, N_tr2, dist="expon")

N_tr4 = 4*N_tr
exp_rv3 = runCDFInversion(N_tr4, avg_lambda1)

print("\t\tPlotting QQ-Plot against quantiles of exponential distribution with average value =", avg_lambda1, "and increasing by 4x the number of exponential draws")
plotQQ(exp_rv3, avg_lambda1, N_tr4, dist="expon")



# --------------------------------------------

