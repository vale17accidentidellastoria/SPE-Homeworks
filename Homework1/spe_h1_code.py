import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st

#---------------------------------------------------
# Settings
#---------------------------------------------------

METRIC = {'median': 'computeMedian', 'mean':'computeMean', 'gap': 'computeGap', 'jain': 'computeJFI', 'stddev': 'computeStdDev', 'variance':'computeVar', 'log_mean':'computeLogMean', 'bernoulli':'bernoulliRVBS'}

#---------------------------------------------------
def loadCSV(csv_file):
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

def computeEta(ci_value, data, distribution):
    if distribution == 't':
        eta = st.t.ppf((1 + ci_value) / 2, len(data) - 1)
    elif distribution == 'normal':
        eta = st.norm.ppf((1 + ci_value) / 2)
    return eta

def computeMedian(x):
    median = 0
    x.sort()
    n = len(x)
    if n % 2 == 0: #means that n is an even number
        median = 0.5 * (x[int(0.5*n) - 1] + x[(int(0.5*n)) + 1 - 1]) #added -1 to conform to the indexes of the arrays
        if(median % 1 == 0): #to convert the median to integer in case we deal with integer values
            median = int(median)
    else: #means that n is an odd number
        median = x[int(0.5*(n+1)) - 1] #added -1 to conform to the indexes of the arrays
    return median

def computeMean(x):
    sum_x = 0
    n = len(x)
    for i in range(0,n):
        sum_x += x[i]
    mean = sum_x / n
    return mean

def computeStdDev(x):
    mean = computeMean(x)
    sum_squares = 0
    n = len(x)
    for i in range(0,n):
        sum_squares += pow((x[i] - mean), 2)
    std_dev = math.sqrt(sum_squares / (n - 1))
    return std_dev

def getCIMedian(data, ci_value):
    data.sort()
    eta = computeEta(ci_value, data, 't') # version for t distribution
    #eta = computeEta(ci_value, data, 'normal') # version for normal distribution
    n = len(data)
    j = int(math.floor(0.5*n - eta*math.sqrt(0.5*n*(1-0.5))))
    k = int(math.ceil(0.5*n + eta*math.sqrt(0.5*n*(1-0.5)))) + 1
    start_int = j+1
    end_int = k-1
    return data[start_int-1], data[end_int-1]

def getCIMean(data, ci_value):
    eta = computeEta(ci_value, data, 't') # version for t distribution
    #eta = computeEta(ci_value, data, 'normal') # version for normal distribution

    #np_mean = np.mean(data) #numpy mean
    mean = computeMean(data)
    #np_sem = st.sem(data) #scipy stats standard error mean
    std_error_mean = computeStdDev(data) / math.sqrt(len(data))
    conf_interval = eta * std_error_mean

    start_int = mean - conf_interval
    end_int = mean + conf_interval

    return start_int, end_int


def computeVar(x, mean):
    std_dev = computeStdDev(x, mean)
    return std_dev**2

def countIntervals(data, i, j):
    data = data[1:]  # remove the first row
    num_intervals = 0
    for x in data:
        mean = computeMean(x)
        if (mean >= i) and (mean <= j):
            num_intervals += 1
    return num_intervals

def computeCov(std_dev, mean):
    coV = std_dev / mean
    return coV

def computeMAD(x, mean):
    sum_abs = 0
    n = len(x)
    for i in range(0,n):
        sum_abs += abs(x[i] - mean)
    mad = sum_abs / n
    return mad

def computeGap(x):
    mean = computeMean(x)
    mad = computeMAD(x, mean)
    gap = mad / (2 * mean)
    return gap

def computeJFI(x):
    sum1 = 0
    sum2 = 0
    n = len(x)
    for i in range(0,n): #compute the sum at the numerator
        sum1 += x[i]
    for i in range(0,n): #compute the sum at the denominator
        sum2 += pow(x[i], 2)
    jfi = pow(sum1, 2) / (n * sum2)
    return jfi

def computeLorenzCurvePoints(x, mean):
    p_gap = []
    l_gap = []
    x.sort()
    n = len(x)
    for i in range(1, n+1):
        p_gap.append(i / n)
        sum_x = 0
        for i2 in range(0,i):
            sum_x += x[i2]
        l_gap.append(sum_x / (mean * n))
    return p_gap, l_gap

def printLorenzCurveGap(p_points, l_points):
    plt.plot(p_points, l_points, linewidth=2.0)
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(x_lim, y_lim, 'k-', color='r', linewidth=1.0)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.show()

#---------------------------------------------------
# funzioni aggiunte da Alberto
#---------------------------------------------------

def bootstrapAlgorithm(dataset, accuracy=25, ci_level=0.95, metric='mean'):
    ds_length = len(dataset)
    samples_metric = []

    samples_metric.append(globals()[METRIC[metric]](dataset))
    R = math.ceil(2 * (accuracy / (1-ci_level))) - 1

    for r in range(R):
        tmp_dataset = []
        for i in range(ds_length):
            tmp_dataset.append(dataset[random.randrange(0, ds_length, 1)])
        samples_metric.append(globals()[METRIC[metric]](tmp_dataset)) # load the desired metric function

    samples_metric.sort()
    #print('sample_metric_len:', len(samples_metric), 'range len:', len(samples_metric[accuracy:(R+1-accuracy)]))
    return samples_metric[accuracy:(R+1-accuracy)]

def printBootsrapMetric(metric):
    plt.plot(metric, np.zeros_like(metric), '.', linewidth=2.0)
    plt.show()

def computeLogMean(x):
    prod = 1
    n = len(x)
    for i in range(0,n):
        prod *= x[i]
    mean = prod**(1/n)

    return mean

def bernoulliRV(dataset):
    counter = 0
    n = len(dataset)
    for i in dataset:
        if i == 1:
            counter += 1
    return counter/n, 1 - (counter/n)

def bernoulliRVBS(dataset):
    return bernoulliRV(dataset)[0]

def bernoulliVar(p):
    return p(1-p)

#---------------------------------------------------

def printCIBootstrap(interval):
    n = len(interval)
    output = "[ " + str(interval[0]) + ", " + str(interval[n-1]) + " ]"
    return output


print("\nExercise 1")

data1 = loadCSV("data_hw1/data_ex1.csv")

median1 = computeMedian(data1)
print("\t 1. The Median is", median1)
start_ci_median, end_ci_median = getCIMedian(data1, 0.95)
print("\t\tThe 95% CI for the Median is [", start_ci_median, ",", end_ci_median, "]")

mean1 = computeMean(data1)

print("\t 2. The Mean is", mean1)
start_ci_mean1_95, end_ci_mean1_95 = getCIMean(data1, 0.95)
start_ci_mean1_99, end_ci_mean1_99 = getCIMean(data1, 0.99)
print("\t\tThe 95% CI for the Mean is [", start_ci_mean1_95, ",", end_ci_mean1_95, "]")
print("\t\tThe 99% CI for the Mean is [", start_ci_mean1_99, ",", end_ci_mean1_99, "]")
#printBootsrapMetric(bootstrapAlgorithm(dataset=data1, metric='median'))


print("\n####################")
#---------------------------------------------------

print("\nExercise 2")
data2 = loadCSV("data_hw1/data_ex2.csv")

start_ci_mean2_firstrow_95, end_ci_mean2_firstrow_95 = getCIMean(data2[0], 0.95)
print("\t 1. The 95% CI for the Mean of data of the first row is [", start_ci_mean2_firstrow_95, ",", end_ci_mean2_firstrow_95, "]")
num_intervals_ex2 = countIntervals(data2, start_ci_mean2_firstrow_95, end_ci_mean2_firstrow_95)
print("\t 2. The number of Means that fall inside the Confidence Interval computed for the first row is", num_intervals_ex2)

print("\n####################")
#---------------------------------------------------

print("\nExercise 3")

data3 = loadCSV("data_hw1/data_ex3.csv")
mean3 = computeMean(data3)
std_dev3 = computeStdDev(data3)

coV3 = computeCov(std_dev3, mean3)
print("\t 1. The Coefficient of Variation for the data is", coV3)
gap3 = computeGap(data3)
print("\t\tThe Lorenz Curve Gap for the data is", gap3)
jfi3 = computeJFI(data3)
print("\t\tThe Jain's fairness index for the data is", jfi3)

print("\t 2. Lorenz Curve Gap plotted!")
p, l = computeLorenzCurvePoints(data3, mean3)
printLorenzCurveGap(p, l)

print("\t 3. Using Bootstrap Algorithm:")
gap_bootstrap95 = bootstrapAlgorithm(dataset=data3, ci_level=0.95, metric='gap')
jain_bootstrap95 = bootstrapAlgorithm(dataset=data3, ci_level=0.95, metric='jain')
mean_bootstrap95 = bootstrapAlgorithm(dataset=data3, ci_level=0.95, metric='mean')
stddev_bootstrap95 = bootstrapAlgorithm(dataset=data3, ci_level=0.95, metric='stddev')
print("\t\tThe 95% CI for Lorentz curve gap is", printCIBootstrap(gap_bootstrap95))
print("\t\tThe 95% CI for Jain’s fairness index is", printCIBootstrap(jain_bootstrap95))
print("\t\tThe 95% CI for the mean of the data is", printCIBootstrap(mean_bootstrap95))
print("\t\tThe 95% CI for the standard deviation of the data is", printCIBootstrap(stddev_bootstrap95))

print("\t\t----------------------")

gap_bootstrap99 = bootstrapAlgorithm(dataset=data3, ci_level=0.99, metric='gap')
jain_bootstrap99 = bootstrapAlgorithm(dataset=data3, ci_level=0.99, metric='jain')
mean_bootstrap99 = bootstrapAlgorithm(dataset=data3, ci_level=0.99, metric='mean')
stddev_bootstrap99 = bootstrapAlgorithm(dataset=data3, ci_level=0.99, metric='stddev')
print("\t\tThe 99% CI for Lorentz curve gap is", printCIBootstrap(gap_bootstrap99))
print("\t\tThe 99% CI for Jain’s fairness index is", printCIBootstrap(jain_bootstrap99))
print("\t\tThe 99% CI for the mean of the data is", printCIBootstrap(mean_bootstrap99))
print("\t\tThe 99% CI for the standard deviation of the data is", printCIBootstrap(stddev_bootstrap99))

print("\t 4. Using Asymptotic formulas:")
start_mean3_asymp95, end_mean3_asymp95 = getCIMean(data3, 0.95)
print("\t\tThe 95% CI for the Mean is [", start_mean3_asymp95, ",", end_mean3_asymp95, "]")
start_mean3_asymp99, end_mean3_asymp99 = getCIMean(data3, 0.99)
print("\t\tThe 99% CI for the Mean is [", start_mean3_asymp99, ",", end_mean3_asymp99, "]")

print("\n####################")
#---------------------------------------------------

print("\nExercise 4")

print("\t 1. Using Bootstrap Algorithm:")
data4 = loadCSV("data_hw1/data_ex4.csv")
bs_95 = bootstrapAlgorithm(dataset=data4)
bs_99 = bootstrapAlgorithm(dataset=data4, ci_level=0.99)

print('\t\tThe 95% CI for mean is [{}, {}]'.format(bs_95[0], bs_95[len(bs_95)-1]))
print('\t\tThe 99% CI for mean is [{}, {}]'.format(bs_99[0], bs_99[len(bs_99)-1]))

print("\t 2. Using Asymptotic formulas:")
start_ci_mean4_95, end_ci_mean4_95 = getCIMean(data4, 0.95)
start_ci_mean4_99, end_ci_mean4_99 = getCIMean(data4, 0.99)

print('\t\tThe 95% CI for mean is [{}, {}]'.format(start_ci_mean4_95, end_ci_mean4_95))
print('\t\tThe 99% CI for mean is [{}, {}]'.format(start_ci_mean4_99, end_ci_mean4_99))

print("\t 3. CI Intervals using log-transformation:")
log_transformed_data4 = [math.log(x+1) for x in data4]
log_bs_95 = bootstrapAlgorithm(dataset=log_transformed_data4, metric='log_mean')
print('\t\tThe 95% CI with bootstrap for log transformation mean is [{}, {}]'.format(log_bs_95[0], log_bs_95[len(log_bs_95)-1]))

'''
# matplotlib histogram
plt.hist(data4, color = 'blue', edgecolor = 'black', bins = 20)

# Add labels
plt.title('Histogram of Data4')
plt.xlabel('density')
plt.ylabel('values')
plt.show()

printBootsrapMetric(bs_95)
'''

#printBootsrapMetric()

#solution for Exercise 4

print("\n####################")
#---------------------------------------------------

print("\nExercise 5")

data5 = loadCSV("data_hw1/data_ex5.csv")

print("\t 1. Probability of Success and CI Using Bootstrap Algorithm:")
p, notp = bernoulliRV(data5)
bernoulli_bs_95 = bootstrapAlgorithm(dataset=data5, metric='bernoulli')
bernoulli_bs_99 = bootstrapAlgorithm(dataset=data5, ci_level=0.99, metric='bernoulli')

print('\t\tThe probability of success is {}.'.format(p))
print('\t\tThe 95% Bernoulli CI with bootstrap is [{}, {}]'.format(bernoulli_bs_95[0], bernoulli_bs_95[len(bernoulli_bs_95)-1]))
print('\t\tThe 99% Bernoulli CI with bootstrap is [{}, {}]'.format(bernoulli_bs_99[0], bernoulli_bs_99[len(bernoulli_bs_99)-1]))

print("\t 2. Finding confidence interval with the Rule of Three:")
dataset5_partial = data5[:15]
print('\t\tdataset5_partial of length {}: {}'.format(len(dataset5_partial), dataset5_partial))
print('\t\tWith the rule of three the CI is [{}, {}]'.format(0, 3/len(dataset5_partial)))


#solution for Exercise 5

print("\n####################")
#---------------------------------------------------