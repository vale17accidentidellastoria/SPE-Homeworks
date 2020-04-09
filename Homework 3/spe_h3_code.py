import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import statsmodels.api as sm

random.seed()


def loadCSV(csv_file):  # TODO: Remember to modify it according to data of exercise 3
    dataset = pd.read_csv(csv_file, header=None)  # here we are working with Pandas DataFrame
    # print(dataset.shape)

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


def makeTransitionProbabilityMatrix():
    tr_matrix = pd.DataFrame({1: [0.75, 0.25, 0, 0],
                              2: [0.25, 0.50, 0.25, 0],
                              3: [0, 0.40, 0.40, 0.20],
                              4: [0, 0, 0.25, 0.75]},
                             index=[1, 2, 3, 4])
    return tr_matrix


def runCDFInversionDiscrete(prob):
    U = random.uniform(0, 1)
    state = 0

    if U < sum(prob[:1]):
        state = 1
    elif (sum(prob[:1])) <= U < (sum(prob[:2])):
        state = 2
    elif (sum(prob[:2])) <= U < (sum(prob[:3])):
        state = 3
    elif (sum(prob[:3])) <= U < (sum(prob[:4])):
        state = 4

    return state


def checkState(d, state_num):
    if state_num == 1:
        d['state1'] += 1
    elif state_num == 2:
        d['state2'] += 1
    elif state_num == 3:
        d['state3'] += 1
    elif state_num == 4:
        d['state4'] += 1
    return d


def markovChain(P, iterations, counters):
    states_history = [] # to keep track of the states visited (history of states): IF NECESSARY...

    initial_state = random.randrange(1, len(P) + 1)  # Set initial state at random (we have 4 states to consider in this exercise)
    states_history.append(initial_state)
    counters = checkState(counters, initial_state)

    next_state = runCDFInversionDiscrete(P[initial_state])

    for i in range(1, iterations):
        states_history.append(next_state)
        counters = checkState(counters, next_state)
        next_state = runCDFInversionDiscrete(P[next_state])

    output_res = {}

    for index, d in enumerate(counters.values()):
        output_res[index+1] = d / iterations

    return counters, output_res, states_history


def computeAvgThroughput(markov_outs, throughputs):
    res = np.sum(np.multiply(np.array(list(markov_outs.values())), throughputs))
    return res


# --------------------------------------------

# Exercise 1

# --------------------------------------------

# Exercise 2

print("\nExercise 2")

MARKOV_ITERS = int(10E4)
intial_states_counter = {'state1': 0, 'state2': 0, 'state3': 0, 'state4': 0}

P = makeTransitionProbabilityMatrix()
# print(P)
state_counter, m_output, history = markovChain(P, MARKOV_ITERS, intial_states_counter)
#print(state_counter)
#print(m_output)

for i in range (1, len(m_output) + 1):
    if i == 1:
        print("\t 1. # times the chain is in State", i ,": is", int(m_output[i]*MARKOV_ITERS) , "/", MARKOV_ITERS, "=", m_output[i])
    else:
        print("\t\t# times the chain is in State", i ,": is", int(m_output[i]*MARKOV_ITERS) , "/", MARKOV_ITERS, "=", m_output[i])

#TODO: write here plots for Point 2 of Exercise 2

print("\t 2. Plotted computed fractions as a function of the number of steps!")

throughput_values = [1500, 1000, 250, 50]
avg_throughput = computeAvgThroughput(m_output, throughput_values)

print("\t 3. The estimated average throughput over the wireless channel is", avg_throughput)

#TODO: write here plots for Exercise 3

print("\n####################")

# --------------------------------------------

# Exercise 3

# --------------------------------------------

# Exercise 4

print("\nExercise 4")

print("\n####################")

# --------------------------------------------
