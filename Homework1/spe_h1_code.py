import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st

#---------------------------------------------------
# Settings
#---------------------------------------------------
METRIC = {'median': 'computeMedian', 'mean':'computeMean'}
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
    sum_x = 0
    n = len(x)
    for i in range(0,n):
        sum_x += x[i]
    mean = sum_x / n
    return mean

def computeStdDev(x, mean):
    sum_squares = 0
    n = len(x)
    for i in range(0,n):
        sum_squares += pow((x[i] - mean), 2)
    std_dev = math.sqrt(sum_squares / (n - 1))
    return std_dev

def getCIMedian(data, ci_value):
    eta = st.t.ppf((1 + ci_value) / 2, len(data) - 1)
    n = len(data)
    j = int(round(0.5*n - eta*math.sqrt(0.5*n*(1-0.5))))
    k = int(round(0.5*n + eta*math.sqrt(0.5*n*(1-0.5)))) + 1
    return data[j-1], data[k-1]

def getCIMean(data, ci_value):
    eta = st.t.ppf((1 + ci_value) / 2, len(data) - 1)
    #np_mean = np.mean(data) #numpy mean
    mean = computeMean(data)
    #np_sem = st.sem(data) #scipy stats standard error mean
    std_error_mean = computeStdDev(data, mean) / math.sqrt(len(data))

    conf_interval = eta * std_error_mean

    start_int = mean - conf_interval
    end_int = mean + conf_interval

    return start_int, end_int

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
        samples_metric.append(globals()[METRIC[metric]](tmp_dataset))
    samples_metric.sort()
    print('sample_metric_len:', len(samples_metric), 'range len:', len(samples_metric[accuracy:(R+1-accuracy)]))
    return samples_metric[accuracy:(R+1-accuracy)]

def printBootsrapMetric(metric):
    plt.plot(metric, np.zeros_like(metric), '.', linewidth=2.0)
    plt.show()


#---------------------------------------------------

print("\nExercise 1")

data1 = loadCSV("data_hw1/data_ex1.csv")
median1 = computeMedian(data1)
print("\t The Median is", median1)
mean1 = computeMean(data1)
print("\t The Mean is", mean1)
printBootsrapMetric(bootstrapAlgorithm(dataset=data1, metric='median'))

start_ci_median, end_ci_median = getCIMedian(data1, 0.95)
print("\t The 95% CI for the Median is between [", start_ci_median, ",", end_ci_median, "]")

start_ci_mean95, end_ci_mean95 = getCIMean(data1, 0.95)
start_ci_mean99, end_ci_mean99 = getCIMean(data1, 0.99)
print("\t The 95% CI for the Mean is between [", start_ci_mean95, ",", end_ci_mean95, "]")
print("\t The 99% CI for the Mean is between [", start_ci_mean99, ",", end_ci_mean99, "]")
#TODO: review the prints of the intervals

print("\n####################")
#---------------------------------------------------

print("\nExercise 2")
data2 = loadCSV("data_hw1/data_ex2.csv")
#print(data2)

#solution for Exercise 2

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


p, l = computeLorenzCurvePoints(data3, mean3)
printLorenzCurveGap(p, l)


print("\n####################")
#---------------------------------------------------

print("\nExercise 4")

data4 = loadCSV("data_hw1/data_ex4.csv")

#solution for Exercise 4

print("\n####################")
#---------------------------------------------------

print("\nExercise 5")

data5 = loadCSV("data_hw1/data_ex5.csv")

#solution for Exercise 5

print("\n####################")
#---------------------------------------------------