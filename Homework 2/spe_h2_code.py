import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
import scipy.stats as st

def loadCSV(csv_file): #TODO: Remember to modify it according to data of exercise 2
    dataset = pd.read_csv(csv_file, header=None) #here we are working with Pandas DataFrame
    #print(dataset.shape)
    data_points = []
    column_dim = dataset.shape[1] #save the number of columns
    row_dim = dataset.shape[0] #save the number of columns
    if (column_dim == 1): #check if we have data organized in a single column
        data_points = dataset[0].values.tolist() #convert the values of the column into a list of values
    elif (column_dim > 1): #check if data is expressed as a matrix of values (multiple columns)
        data_points = dataset.values.tolist()
        if(row_dim == 1): #check if data is expressed as a single row
            data_points = data_points[0]
    return data_points

def computeBins(data):
    """ Computes the necessary number of bins for the histogram according to the input data """
    num_observation = len(data)  # the number of observations
    data_range = max(data) - min(data)  # range is the difference between minimum value and maximum value
    num_intervals = int(round(math.sqrt(num_observation)))
    binwidth = data_range / num_intervals
    return np.arange(min(data), max(data) + binwidth, binwidth)

def plotHistogram(data):
    bins_hist = computeBins(data)
    plt.hist(data, bins=bins_hist, density=True)
    plt.show()

def plotPDFsOnHistogram(data, mu_gauss, sigma_gauss):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    bins_hist = computeBins(data)
    ax1.hist(data, bins=bins_hist, density=True, alpha=0.8, color='g')

    ax2 = ax1.twinx()

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

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)

def EMalgorithm(data, n, update_prior_belief=False):
    num_iterations = 100
    d_length = len(data)
    p_gauss, mu_gauss, sigma_gauss = setPriorBelief(n)
    for i in range(0, num_iterations):
        if (i >= 1) and update_prior_belief:
            p_gauss = updatePriorBelief(gauss, d_length, n)
        p_x_gauss = runExpectationStep(data, n, mu_gauss, sigma_gauss)
        gauss = setPosteriorBelief(d_length, p_gauss, p_x_gauss)
        mu_gauss, sigma_gauss = runMaximizationStep(data, n, gauss, mu_gauss, sigma_gauss)

    return mu_gauss, sigma_gauss

def setPriorBelief(n):
    m = np.array([1/3, 1/3, 1/3])  # We deal with three distributions
    p_gauss = m / np.sum(m)
    mu_gauss = [5, -4, 2]
    sigma_gauss = [4, 3, 6]
    return p_gauss, mu_gauss, sigma_gauss

def updatePriorBelief(gauss, length, n):
    p_g = []
    for i in range(0,n):
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
        den = (p_x_gauss[0][i]*p_gauss[0] + p_x_gauss[1][i]*p_gauss[1] + p_x_gauss[2][i]*p_gauss[2])
        gauss1.append((p_x_gauss[0][i]*p_gauss[0]) / den)
        gauss2.append((p_x_gauss[1][i]*p_gauss[1]) / den)
        gauss3.append((p_x_gauss[2][i]*p_gauss[2]) / den)

    return [gauss1, gauss2, gauss3]

def runMaximizationStep(data, n, gauss, mu_gauss, sigma_gauss):
    for i in range(0, n):
        mu_gauss[i] = np.sum(np.multiply(gauss[i], data)) / np.sum(gauss[i])
        diff = [x - mu_gauss[i] for x in data]
        sigma_gauss[i] = math.sqrt(np.sum(np.multiply(gauss[i], np.power(diff, 2))) / np.sum(gauss[i]))

    return mu_gauss, sigma_gauss


#--------------------------------------------


#Exercise 2
data2 = loadCSV("data_hw2/data_ex2.csv")
NUM_GAUSS = 3

plotHistogram(data2)

mu_gauss, sigma_gauss = EMalgorithm(data2, NUM_GAUSS, update_prior_belief=False)
plotPDFsOnHistogram(data2, mu_gauss, sigma_gauss)
print("The means for the 3 Gaussians distributions are:", mu_gauss)
print("The standard deviations for the 3 Gaussians distributions are:", sigma_gauss)

mu_gauss_up_prior, sigma_gauss_up_prior = EMalgorithm(data2, NUM_GAUSS, update_prior_belief=True)
plotPDFsOnHistogram(data2, mu_gauss_up_prior, sigma_gauss_up_prior)
print("The means for the 3 Gaussians distributions (with prior update) are:", mu_gauss_up_prior)
print("The standard deviations for the 3 Gaussians distributions (with prior update) are:", sigma_gauss_up_prior)

#Exercise 4
data4 = [[2, 1], [3, 4], [4, 2], [5, 7], [6, 10], [7, 9], [8, 9], [9, 14], [10, 7], [11, 5], [12, 3]]
num_occurrences = []
for i in range(0, len(data4)):
    num_occurrences.append(data4[i][1])

#plotHistogram(num_occurrences, print_distr=False)

