import pandas as pd
import math
import matplotlib.pyplot as plt

def loadCSV(csv_file):
    dataset = pd.read_csv(csv_file, header=None) #here we are working with Pandas DataFrame
    #print(dataset.shape)
    data_points = []
    column_dim = dataset.shape[1] #save the number of columns
    if (column_dim == 1): #check if we have data organized in a single column
        data_points = dataset[0].values.tolist() #convert the values of the column into a list of values 
    elif (column_dim > 1): #check if data is expressed as a matrix of values (multiple columns)
        data_points = dataset.values.tolist()
    #TODO: check if we have only a single row of values
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

def computeStdDev(x, mean):
    sum_squares = 0
    n = len(x)
    for i in range(0,n):
        sum_squares += pow((x[i] - mean), 2)
    std_dev = math.sqrt(sum_squares / (n -1)) #divide by n or (n-1)??
    return std_dev

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

print("\nExercise 1")

data1 = loadCSV("data_hw1/data_ex1.csv")
median1 = computeMedian(data1)
print("\t The Median is", median1)
mean1 = computeMean(data1)
print("\t The Mean is", mean1)

print("\n####################")
#---------------------------------------------------

print("\nExercise 2")

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

p = []
l = []
computeLorenzCurvePoints(data3, mean3, p, l)
printLorenzCurveGap(p,l)


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