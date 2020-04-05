import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st

COLORS = {'blue':'#0a0add', 'red':'#FF0000', 'black':'#000000'} 

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

def plotHistogram(data, print_distr=False):
    bins_hist = computeBins(data)
    plt.hist(data, bins=bins_hist)
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

#--------------------------------------------

#if __name__ == "__main__":

#Exercise 1
data1 = loadCSV("data_hw2/data_ex1.csv")

#exercise1(data1, poly_grade=[2, 5])
'''
sp = 0
results = []
val = 1000000.0
for i in range(1000000):
    p = bernoulliRNG(0.05)
    results.append(p)
    if p == 1:
        sp += 1
'''

exercise3()
exercise3(n_exp=100, p_success=0.9)
#print('results {}.'.format(results))
#print('Success probability is {}.'.format(sp/val))


'''
## scatte plot of data1
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''

'''
#Exercise 2
data2 = loadCSV("data_hw2/data_ex2.csv")

plotHistogram(data2, print_distr=False)



#Exercise 4
data4 = [[2, 1], [3, 4], [4, 2], [5, 7], [6, 10], [7, 9], [8, 9], [9, 14], [10, 7], [11, 5], [12, 3]]
num_occurrences = []
for i in range(0, len(data4)):
    num_occurrences.append(data4[i][1])

plotHistogram(num_occurrences, print_distr=False)
'''
