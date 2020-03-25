import pandas as pd
import math

def loadCSV(csv_file):
    dataset = pd.read_csv(csv_file, header=None) #here we are working with Pandas DataFrame
    #print(dataset.shape)
    data_points = []
    column_dim = dataset.shape[1] #save the number of columns
    if (column_dim == 1): #check if we have data organized in a single column
        data_points = dataset[0].values.tolist() #convert the values of the column into a list of values 
    elif (column_dim > 1): #check if data is expressed as a matrix of values (multiple columns)
        data_points = dataset.values.tolist()
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
    std_dev = math.sqrt(sum_squares / n)
    return std_dev

data = loadCSV("data_hw1/data_ex1.csv")
print("\nExercise 1")

median = computeMedian(data)
print("\t The Median is", median)

mean = computeMean(data)
print("\t The Mean is", mean)

std_dev = computeStdDev(data, mean)
print("\t The Standard Deviation is", std_dev)

print("\n####################\n")

print("\nExercise 2")