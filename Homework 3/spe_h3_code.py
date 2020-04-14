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
    #plt.show()

def computeSinc(x, A=1.8988):
    if x == 0:
        return 1
    return (1/A)*abs(math.sin(math.pi*x)/(math.pi*x))

def computeSamplingRejection(runs=20000):
    accepted_x=[]
    accepted_y=[]
    rejected_x=[]
    rejected_y=[]
    for run in range(runs):
        x = random.uniform(-6, 6)
        y = random.uniform(0, 1)/1.8988
        if y <= computeSinc(x):
            accepted_x.append(x)
            accepted_y.append(y)
        else:
            rejected_x.append(x)
            rejected_y.append(y)
    return accepted_x, accepted_y, rejected_x, rejected_y

def intervalError(n_s, n_f, n, z=1.96): 
    return (1.96/n)*math.sqrt(n_s*(1-(n_s/float(n))))

class Network():
    def __init__(self, r, n):
        self.r = r
        self.n = n
        self.nodes = np.zeros(shape=(r, n), dtype=int)
        self.k_nodes = np.zeros(shape=(r), dtype=int)
        self.destination = 0

    def propagate(self, p=0.1):
        source = 1
        destination = 0
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                if i == 0:
                    self.nodes[i][j] = np.random.binomial(1, 1-p, 1)
                else:
                    k = sum(self.nodes[i-1])
                    if k == 0:
                        return 0
                    self.nodes[i][j] = np.random.binomial(1, 1-(p**k), 1)
            self.k_nodes[i] = sum(self.nodes[i])
            #print('{}\t| k = {} \t| values = {}'.format(i, sum(self.nodes[i]), self.nodes[i]))

        last_k = sum(self.nodes[len(self.nodes)-1])
        #destination =  np.random.binomial(1, 1-(p**last_k), 1)
        #print('k = {} \t| D: {}.'.format(last_k, destination))
        self.destination = np.random.binomial(1, 1-(p**last_k), 1)[0]

def meanSuccessProb(n_s_p):
    sum = 0
    for p in n_s_p:
        sum += p
    return sum/float(len(n_s_p))

class Case():
    def __init__(self, networks, d_success, ks_success, ie, stages, nodes):
        self.networks = networks
        self.d_success = d_success
        self.ks_success = ks_success
        self.ie = ie
        self.stages = stages
        self.nodes = nodes

def monteCarloSim(p_error=0.1, trials=10000, network_shapes=[[2, 2],[5, 10]]):
    cases = []

    for network in network_shapes:
        networks = []
        d_success = []
        ks_success = []

        for i in range(trials):
            tmp_network = Network(network[0], network[1])
            tmp_network.propagate(p_error)
            networks.append(tmp_network)
            d_success.append(tmp_network.destination)
            ks_success.append(tmp_network.k_nodes)

        ie = intervalError(sum(d_success), trials-sum(d_success), trials)
        tmp_ks = np.array(ks_success)
        ks = []
        for i in range(network[0]):
            ks.append(sum(tmp_ks[:,i]/trials))
        #print(tmp_ks[0], ks)

        #ks_success = intervalError(sum(ks_success), network[1]-sum(ks_success), network[1])
        case = Case(networks, d_success, ks, ie, network[0], network[1])
        cases.append(case)


    '''
    Network = Network(2, 2)
    n_success = 0

    for i in range(trials):
        n_success += Network.propagate(p_error)

    trials = float(trials)

    mean = meanSuccessProb(n_s_p)
    
    ie = intervalError(n_s, n-n_s, n)

    print('Success has {} % probability and error probability of {}.'.format(success/trials, 1-success/trials))


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
    '''

    return cases

def plotBarDiagram(items, probs):
    plt.bar(items, probs, width=0.2)
    plt.show()
# --------------------------------------------

# Exercise 1

print("\nExercise 1")
runs = 50000
results_x = []
results_y = []
for x in np.arange(-6, 7, 0.1):
    y = computeSinc(x)
    #print(x, y)
    results_x.append(x)
    results_y.append(y)

empirical_x, empirical_y, bad_x, bad_y = computeSamplingRejection(runs)
#print(empirical_y)

empirical = (empirical_x, empirical_y)
bad = (bad_x, bad_y)

data = (empirical, bad)
colors = ("blue", "red")
groups = ("accepted", "rejected")


plt.subplot(1, 3, 1)
for data, color, group in zip(data, colors, groups):
    x, y = data
    plt.title('Rejection sampling with {} runs'.format(runs))
    #plt.plot(empirical_x, empirical_y, 'ro', markerfacecolor='blue', markersize=1)
    #plt.plot(bad_x, bad_y, 'ro', markerfacecolor='red', markersize=1)
    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=2, label=group)
    plt.ylabel("Probability")
    plt.xlabel("x")
    plt.legend(loc=2)

#plt.title('Rejection sampling with {} runs'.format(runs))
plt.subplot(1, 3, 2)
plt.title('Empirical PDF')
#plt.bar(empirical_x, empirical_y)
#plotHistogram(empirical_y)
#plt.hist(empirical_y, bins=np.arange(-6, 6.1, 0.1), density=False)
plt.hist(empirical_x, density=True, bins=int(np.around(1+3.3*math.log(runs), 0)), edgecolor='black', linewidth=0.5)
plt.ylabel("Probability")
plt.xlabel("Sampling")

plt.subplot(1, 3, 3)
plt.title('f(x)')
plt.plot(results_x, results_y)
plt.ylabel("Probability")
plt.xlabel("x")
plt.show()

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

data3 = np.array(loadCSV("data_hw3/theory_ex3.csv"))

trials = 500
prob = 0.7
cases = monteCarloSim(p_error=prob, trials=trials)

k_avg = []
k_error = []

for case in cases:
    d_success = sum(case.d_success)
    tmp_k_avg = []
    tmp_k_error = []
    for k in case.ks_success:
        #print('k', k, case.ks_success[1])
        tmp_k_avg.append(k)
        tmp_k_error.append(intervalError(k, case.nodes-k, case.nodes))
    k_avg.append(tmp_k_avg)
    k_error.append(tmp_k_error)
    print('Success has {} % probability and error probability of {}.'.format(d_success/trials, 1-d_success/trials))

p_cases = []
p_errors = data3[:,0]#np.arange(0.02, 1.0, 0.02)

for p in p_errors:
    p_cases.append(monteCarloSim(p_error=p, trials=trials))

y_cases = []
y_errors = []



for p_case in p_cases:
    tmp_y = []
    tmp_error = []
    #tmp_k_avg = []
    for case in p_case:
        #print(1-(sum(case.d_success)/float(trials)), end=";")
        tmp_y.append(1-(sum(case.d_success)/float(trials)))
        tmp_error.append(case.ie)
        #tmp_k_avg.append(case.ks_success)
    y_cases.append(tmp_y)
    y_errors.append(tmp_error)


y = np.array(y_cases)
ie = np.array(y_errors)
#print(p_errors, y[:,0])

plt.subplot(2, 2, 1)
plt.title('Probability of error for r=2 and N=2')
plt.errorbar(p_errors, y[:,0], yerr=ie[:,0], linestyle='None', marker='.')
plt.ylabel("p of error at D")
plt.xlabel("p of error")
plt.subplot(2, 2, 2)
plt.title('Probability of error for r=5 and N=10')
plt.errorbar(p_errors, y[:,1], yerr=ie[:,1], linestyle='None', marker='.')
plt.ylabel("p of error at D")
plt.xlabel("p of error")


plt.subplot(2, 2, 3)
plt.title('Theoretical Probability of error for r=2 and N=2')
plt.errorbar(p_errors, data3[:,1], linestyle='None', marker='.')
plt.ylabel("p of error at D")
plt.xlabel("p of error")
plt.subplot(2, 2, 4)
plt.title('Theoretical Probability of error for r=5 and N=10')
plt.errorbar(p_errors, data3[:,2], linestyle='None', marker='.')
plt.ylabel("p of error at D")
plt.xlabel("p of error")


plt.show()

x_2 = ['1', '2']
x_10 = ['1', '2', '3', '4', '5']
plt.subplot(1, 2, 1)
plt.title('K nodes for r=2 and N=2')
plt.errorbar(x_2, k_avg[0], yerr=k_error[0], linestyle='None', marker='.')
plt.ylabel("average successful nodes")
plt.xlabel("relay")
plt.subplot(1, 2, 2)
plt.title('k nodes for r=5 and N=10')
plt.errorbar(x_10, k_avg[1], yerr=k_error[1], linestyle='None', marker='.')
plt.ylabel("average successful nodes")
plt.xlabel("relay")

plt.show()


# --------------------------------------------

# Exercise 4

print("\nExercise 4")

print("\n####################")

# --------------------------------------------
