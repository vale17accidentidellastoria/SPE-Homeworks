import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st

#---------------------------------------------------
# Settings
#---------------------------------------------------
METRIC = {'median': 'computeMedian', 'mean':'computeMean', 'variance':'computeVar', 'log_mean':'computeLogMean'}
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
    sum = 0
    n = len(x)
    for i in range(0,n):
        sum += x[i]
    mean = sum / n
    return mean

def computeLogMean(x):
    prod = 1
    n = len(x)
    for i in range(0,n):
        prod *= x[i]
    mean = prod**(1/n)
    #if not isinstance(mean, complex):
        #print('mean:', mean, 'prod:', prod)
    return mean

def computeStdDev(x, mean):
    sum_squares = 0
    n = len(x)
    for i in range(0,n):
        sum_squares += pow((x[i] - mean), 2)
    std_dev = math.sqrt(sum_squares / (n -1)) #divide by n or (n-1)??
    return std_dev

def getCIMedian(data, ci_value):
    eta = st.t.ppf((1 + ci_value) / 2, len(data) - 1)
    n = len(data)
    j = int(round(0.5*n - eta*math.sqrt(0.5*n*(1-0.5))))
    k = int(round(0.5*n + eta*math.sqrt(0.5*n*(1-0.5)))) + 1
    return data[j-1], data[k-1]

def getCIMean(data, ci_value):
    eta = st.t.ppf((1 + ci_value) / 2, len(data) - 1)
    print(eta)
    #np_mean = np.mean(data) #numpy mean
    mean = computeMean(data)
    
    #np_sem = st.sem(data) #scipy stats standard error mean
    std_error_mean = computeStdDev(data, mean) / math.sqrt(len(data))
    conf_interval = eta * std_error_mean

    start_int = mean - conf_interval
    end_int = mean + conf_interval

    return start_int, end_int

def computeVar(x, mean):
    std_dev = computeStdDev(x, mean)
    return std_dev**2

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

def computeGap(x, mean):
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

def computeLorenzCurvePoints(x, mean, p, l): #TODO: check if the function is really correct! 
    x.sort()
    n = len(x)
    for i in range(1,n+1):
        p.append(i / n)
        sum = 0
        for i2 in range(0,i):
            sum += x[i2]
        l.append(sum / (mean * n))

def printLorenzCurveGap(p, l):
    plt.plot(p, l, linewidth=2.0)
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(x_lim, y_lim, 'k-', color = 'r', linewidth=1.0)
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

    print('bs mean', samples_metric)

    for r in range(R):
        tmp_dataset = []
        for i in range(ds_length):
            tmp_dataset.append(dataset[random.randrange(0, ds_length, 1)])
        samples_metric.append(globals()[METRIC[metric]](tmp_dataset)) # load the desired metric function

    samples_metric.sort()
    print('sample_metric_len:', len(samples_metric), 'range len:', len(samples_metric[accuracy:(R+1-accuracy)]))
    return samples_metric[accuracy:(R+1-accuracy)]

def printBootsrapMetric(metric):
    plt.plot(metric, np.zeros_like(metric), '.', linewidth=2.0)
    plt.show()

def exercise2(dataset):
    # TODO: compute first row CI 95%
    ci = [3, 10]

    means = []
    for row in dataset[1:]:
        means.append(computeMean(row))
    
    counter_mean = 0
    for mean in means:
        if ci[0] <= mean <= ci[1]:
            counter_mean += 1
    
    print(counter_mean, "out of 6000 means are inside the CI", ci)

#---------------------------------------------------

print("\nExercise 1")

data1 = loadCSV("data_hw1/data_ex1.csv")
median1 = computeMedian(data1)
print("\t The Median is", median1)
mean1 = computeMean(data1)
print("\t The Mean is", mean1)


print("\n####################")
#---------------------------------------------------

print("\nExercise 2")
data2 = loadCSV("data_hw1/data_ex2.csv")

exercise2(data2)

print("\n####################")
#---------------------------------------------------

print("\nExercise 3")

data3 = loadCSV("data_hw1/data_ex3.csv")
mean3 = computeMean(data3)
std_dev3 = computeStdDev(data3, mean3)

coV3 = computeCov(std_dev3, mean3)
print("\t The Coefficient of Variation for the data is", coV3)
gap3 = computeGap(data3, mean3)
print("\t The Lorenz Curve Gap for the data is", gap3)
jfi3 = computeJFI(data3)
print("\t The Jain's fairness index for the data is", jfi3)

p = []
l = []
computeLorenzCurvePoints(data3, mean3, p, l)
printLorenzCurveGap(p,l)


print("\n####################")
#---------------------------------------------------

print("\nExercise 4")

data4 = loadCSV("data_hw1/data_ex4.csv")
bs_95 = bootstrapAlgorithm(dataset=data4)
bs_99 = bootstrapAlgorithm(dataset=data4, ci_level=0.99)

print('95% CI with bootstrap is [{}, {}]'.format(bs_95[0], bs_95[len(bs_95)-1]))
print('99% CI with bootstrap is [{}, {}]'.format(bs_99[0], bs_99[len(bs_99)-1]))

start_ci_mean4_95, end_ci_mean4_95 = getCIMean(data4, 0.95)
start_ci_mean4_99, end_ci_mean4_99 = getCIMean(data4, 0.99)

print('95% CI with asymptotic formulas is [{}, {}]'.format(start_ci_mean4_95, end_ci_mean4_95))
print('99% CI with asymptotic formulas is [{}, {}]'.format(start_ci_mean4_99, end_ci_mean4_99))

log_transformed_data4 = [math.log(x+1) for x in data4]
print('log transformation', log_transformed_data4[:5])
for e in log_transformed_data4:
    if e < 0:
        print(e)
log_mean = computeLogMean(log_transformed_data4)
print('log_mean =', log_mean)

log_bs_95 = bootstrapAlgorithm(dataset=log_transformed_data4, metric='log_mean')
print('Log mean with 95% CI with bootstrap is [{}, {}]'.format(log_bs_95[0], log_bs_95[len(log_bs_95)-1]))

#printBootsrapMetric()

#solution for Exercise 4

print("\n####################")
#---------------------------------------------------

print("\nExercise 5")

data5 = loadCSV("data_hw1/data_ex5.csv")

#solution for Exercise 5

print("\n####################")
#---------------------------------------------------