import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import statsmodels.api as sm
from scipy.spatial import distance

random.seed()

METRIC = {'median': 'computeMedian', 'mean': 'computeMean', 'gap': 'computeGap', 'jain': 'computeJFI',
          'stddev': 'computeStdDev', 'variance': 'computeVar', 'log_mean': 'computeLogMean',
          'bernoulli': 'bernoulliRVBS'}


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
    states_history = []  # to keep track of the states visited (history of states): IF NECESSARY...

    initial_state = random.randrange(1, len(
        P) + 1)  # Set initial state at random (we have 4 states to consider in this exercise)
    states_history.append(initial_state)
    counters = checkState(counters, initial_state)

    next_state = runCDFInversionDiscrete(P[initial_state])

    for i in range(1, iterations):
        states_history.append(next_state)
        counters = checkState(counters, next_state)
        next_state = runCDFInversionDiscrete(P[next_state])

    output_res = {}

    for index, d in enumerate(counters.values()):
        output_res[index + 1] = d / iterations

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

    return cases

def plotBarDiagram(items, probs):
    plt.bar(items, probs, width=0.2)
    plt.show()


def computeMean(x):
    sum_x = 0
    n = len(x)
    for i in range(0, n):
        sum_x += x[i]
    mean = sum_x / n
    return mean


def bootstrapAlgorithm(dataset, accuracy=25, ci_level=0.95, metric='mean'):
    ds_length = len(dataset)
    samples_metric = []

    samples_metric.append(globals()[METRIC[metric]](dataset))
    R = math.ceil(2 * (accuracy / (1 - ci_level))) - 1

    for r in range(R):
        tmp_dataset = []
        for i in range(ds_length):
            tmp_dataset.append(dataset[random.randrange(0, ds_length, 1)])
        samples_metric.append(globals()[METRIC[metric]](tmp_dataset))  # load the desired metric function

    samples_metric.sort()
    # print('sample_metric_len:', len(samples_metric), 'range len:', len(samples_metric[accuracy:(R+1-accuracy)]))
    return samples_metric[accuracy:(R + 1 - accuracy)]


def plotTimeSlotMarkov(states, num_point):
    x_axis = np.arange(1, num_point + 1)
    s = states[:num_point]

    y_axis_state1 = []
    y_axis_state2 = []
    y_axis_state3 = []
    y_axis_state4 = []

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    for i in range(0, len(s)):
        if s[i] == 1:
            count_1 += 1
        elif s[i] == 2:
            count_2 += 1
        elif s[i] == 3:
            count_3 += 1
        elif s[i] == 4:
            count_4 += 1
        y_axis_state1.append(count_1 / (i + 1))
        y_axis_state2.append(count_2 / (i + 1))
        y_axis_state3.append(count_3 / (i + 1))
        y_axis_state4.append(count_4 / (i + 1))

    fig = plt.figure()
    ax = fig.gca()
    ax.set_yticks(np.arange(0, 1.5, 0.1))

    plt.scatter(x_axis, y_axis_state1, marker='.', color='b', linewidths=1)
    plt.scatter(x_axis, y_axis_state2, marker='.', color='r', linewidths=1)
    plt.scatter(x_axis, y_axis_state3, marker='.', color='g', linewidths=1)
    plt.scatter(x_axis, y_axis_state4, marker='.', color='m', linewidths=1)
    plt.grid(b=None, axis='y')
    plt.xlabel("# of state transitions")
    plt.show()


def setThroughputForStates(s, tr_values):
    s_tr = []

    for i in range(0, len(s)):
        if s[i] == 1:
            s_tr.append(tr_values[0])
        elif s[i] == 2:
            s_tr.append(tr_values[1])
        if s[i] == 3:
            s_tr.append(tr_values[2])
        elif s[i] == 4:
            s_tr.append(tr_values[3])

    return s_tr


def plotAvgThroughput(s, n_points, throughputs):
    x_axis = np.arange(1, n_points + 1)

    s = s[:n_points]

    states_tr = setThroughputForStates(s, throughputs)

    y_axis = []

    for i in range(0, len(states_tr)):
        index = i + 1
        y_axis.append(sum(states_tr[:index]) / index)

    C_I = bootstrapAlgorithm(states_tr, ci_level=0.95, metric='mean')

    fig, ax1, = plt.subplots()
    ax1.scatter(x_axis, y_axis, marker='.', color='b', linewidths=1)
    plt.axhspan(C_I[0], C_I[-1], color='b', alpha=0.2)
    plt.xlabel("# of state transitions")
    plt.ylabel("Throughput (Mbit/s)")
    plt.show()


def createRandomCoordinates(xi, yi, xj, yj, num=1000):
    coord_i = []
    coord_j = []

    for i in range(num):
        x_i = int(random.uniform(xi[0], xi[1] + 1))
        y_i = int(random.uniform(yi[0], yi[1] + 1))
        x_j = int(random.uniform(xj[0], xj[1] + 1))
        y_j = int(random.uniform(yj[0], yj[1] + 1))
        coord_i.append((x_i, y_i))
        coord_j.append((x_j, y_j))

    return coord_i, coord_j


def computeAreas(x_i, y_i, x_j, y_j):
    l1_i = x_i[1] - x_i[0]
    l2_i = y_i[1] - y_i[0]
    Ai = l1_i * l2_i

    l1_j = x_j[1] - x_j[0]
    l2_j = y_j[1] - y_j[0]
    Aj = l1_j * l2_j

    return Ai, Aj

def generateExpRV(exp_lambda):
    U = random.uniform(0, 1)
    x = - (math.log(U) / exp_lambda)
    return x


def runCDFInversionExp(N, avg_lambda):
    rvs_exp = []
    for i in range(0, N):
        rvs_exp.append(generateExpRV(avg_lambda))
    return rvs_exp


def computeSNR(fading_coeff, dist):
    k = 2
    p_n = 3.2E-5
    p_t = 5
    snr = (fading_coeff * p_t * pow(dist, -k)) / p_n
    return snr


def indicatorFunction(snr, threshold):
    if snr < threshold:
        return 1
    else:
        return 0


def monteCarloIntegration(num_draws, num_draws_exp, coords_i, coords_j, theta, A_i, A_j):
    res = []
    for i in range(0, num_draws):
        dist = distance.euclidean(coords_i[i], coords_j[i])

        random_fadings = runCDFInversionExp(num_draws_exp, 1)

        for ind in range(0, len(random_fadings)):
            fading = random_fadings[ind]
            snr = computeSNR(fading, dist)
            indicator = indicatorFunction(snr, theta)

            tmp_p = (math.exp(-fading) / (A_i * A_j)) * indicator

            res.append(tmp_p)

    return np.sum(res) / len(res)


# --------------------------------------------

# Exercise 1

print("\nExercise 1")
print("\t #1 and #2 graph")
runs = 50000
results_x = []
results_y = []
for x in np.arange(-6, 7, 0.1):
    y = computeSinc(x)
    #print(x, y)
    results_x.append(x)
    results_y.append(y)

empirical_x, empirical_y, bad_x, bad_y = computeSamplingRejection(runs)

empirical = (empirical_x, empirical_y)
bad = (bad_x, bad_y)

data = (empirical, bad)
colors = ("blue", "red")
groups = ("accepted", "rejected")


plt.subplot(1, 3, 1)
for data, color, group in zip(data, colors, groups):
    x, y = data
    plt.title('Rejection sampling with {} runs'.format(runs))
    plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=2, label=group)
    plt.ylabel("Probability")
    plt.xlabel("x")
    plt.legend(loc=2)

plt.subplot(1, 3, 2)
plt.title('Empirical PDF')
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
state_counter, m_output, history = markovChain(P, MARKOV_ITERS, intial_states_counter)

for i in range(1, len(m_output) + 1):
    if i == 1:
        print("\t 1. # times the chain is in State", i, ": is", int(m_output[i] * MARKOV_ITERS), "/", MARKOV_ITERS, "=",
              m_output[i])
    else:
        print("\t\t# times the chain is in State", i, ": is", int(m_output[i] * MARKOV_ITERS), "/", MARKOV_ITERS, "=",
              m_output[i])

num_points = int(10E2)
plotTimeSlotMarkov(history, num_points)
print("\t 2. Plotted computed fractions as a function of the number of steps!")

throughput_values = [1500, 1000, 250, 50]
avg_throughput = computeAvgThroughput(m_output, throughput_values)
print("\t 3. The estimated average throughput over the wireless channel is", avg_throughput)

states_and_thr = setThroughputForStates(history, throughput_values)
ci = bootstrapAlgorithm(states_and_thr, ci_level=0.95, metric='mean')
print("\t\tThe 95% CI for the average throughput is between [", ci[0], ",", ci[-1], "]")

n_points_throughput = int(10E2)
plotAvgThroughput(history, n_points_throughput, throughput_values)

print("\n####################")


# --------------------------------------------

# Exercise 3
print("\nExercise 3")
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
    print('\tSuccess has {} % probability and error probability of {}.'.format(d_success/trials, 1-d_success/trials))

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

theta_values = np.arange(1, 320 + 1)

num_position_draws = 1000
num_fading_draws = 50

#coordinates_i, coordinates_j = createRandomCoordinates(num=num_position_draws)
xi_lim = (0,20)
yi_lim = (0,60)
xj_lim = (60,80)
yj_lim = (0,60)
coordinates_i, coordinates_j = createRandomCoordinates(xi_lim, yi_lim, xj_lim, yj_lim, num=num_position_draws)
A_i, A_j = computeAreas(xi_lim, yi_lim, xj_lim, yj_lim)

probs_values = []

for th in range(1, len(theta_values) + 1):
    probs_values.append(monteCarloIntegration(num_position_draws, num_fading_draws, coordinates_i, coordinates_j, th, A_i, A_j))

print("\t 1. The probabilities computed for each value of theta between [1, 320] are:")
print("\t\t" + str(probs_values))

plt.plot(theta_values, probs_values)
plt.show()
print("\t\tPlotted variation of p vs theta!")

#-------------

new_coordinates_i, new_coordinates_j = createRandomCoordinates((10,40), (0,10), (20,60), (20,90), num=num_position_draws)
new_A_i, new_A_j = computeAreas((10,40), (0,10), (20,60), (20,90))

new_probs_values = []

for th in range(1, len(theta_values) + 1):
    new_probs_values.append(monteCarloIntegration(num_position_draws, num_fading_draws, new_coordinates_i, new_coordinates_j, th, new_A_i, new_A_j))

plt.plot(theta_values, new_probs_values)
plt.show()
print("\t 2. Plot obtained by playing with the size of the area of positions i and j!")

#-------------

new_num_position_draws1 = 500
new_num_fading_draws1 = 20

probs_values = []

for th in range(1, len(theta_values) + 1):
    probs_values.append(monteCarloIntegration(new_num_position_draws1, new_num_fading_draws1, coordinates_i, coordinates_j, th, A_i, A_j))

plt.plot(theta_values, probs_values)
plt.show()
print("\t\tPlotted with reduction in the number of realizations (", new_num_position_draws1, "for node positions and", new_num_fading_draws1, "for fading )!")

print("\n####################")

#-------------

new_num_position_draws2 = 100
new_num_fading_draws2 = 20

probs_values = []

for th in range(1, len(theta_values) + 1):
    probs_values.append(monteCarloIntegration(new_num_position_draws2, new_num_fading_draws2, coordinates_i, coordinates_j, th, A_i, A_j))

plt.plot(theta_values, probs_values)
plt.show()
print("\t\tPlotted with reduction in the number of realizations (", new_num_position_draws2, "for node positions and", new_num_fading_draws2, "for fading )!")

print("\n####################")

# --------------------------------------------
