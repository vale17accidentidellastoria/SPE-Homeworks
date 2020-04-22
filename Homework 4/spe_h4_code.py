import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import statsmodels.api as sm
from scipy.spatial import distance
import heapq
import enum

random.seed()

MAX_SIMULATION_TIME = 1000

DEBUG = False
PRINT_SPEED = False
PRINT_NODES = False

class EType(enum.Enum):
    """
    Enumerator class that defines the types of events
    """
    start = "START"
    end = "END"
    arrival = "ARRIVAL"
    departure = "DEPARTURE"
    debug = "DEBUG"
    waypoint_reached = "WAYPOINTREACHED"
    measure_speed = "MEASURESPEED"



class Event:
    """
    Class that defines an event with the three attributes: occurrence time, event type, pointer to the next event
    """
    def __init__(self, occurrence_time, e_type, next_event):
        self.occurrence_time = occurrence_time
        self.e_type = e_type
        self.next_event = next_event


class Queue:
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def __iter__(self):
        return iter(self.queue)

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for adding (enqueue) an element in the queue
    def enqueue(self, event):
        if self.isEmpty():
            self.queue.append(event)
        else:
            self.queue.append(event)
            self.queue.sort(key=lambda x: x.occurrence_time, reverse=False) # order events according to their occurrence time
            events = iter(self.queue) # iterates over all the events contained in the queue
            next(events)
            for elem in self.queue: # add for each event a pointer to the next event
                elem.next_event = next(events, None) # None as a default value in case there is not a "next" element in the iterator

    # for removing (dequeue) the first element in the queue
    def dequeue(self):
        e_pop = None
        if not self.isEmpty():
            self.queue.sort(key=lambda x: x.occurrence_time, reverse=True)
            e_pop = self.queue.pop()
            self.queue.sort(key=lambda x: x.occurrence_time, reverse=False)
        return e_pop


class Simulator:
    def __init__(self):
        self.event_queue = Queue() # initialize the event queue
        self.current_time = 0 # set the current time to 0
        self.max_simulation_time = MAX_SIMULATION_TIME # set a maximum simulation time
        self.initStartEndEventsQueue() # to insert the start event and end event in the queue

    def initStartEndEventsQueue(self):
        start_event = Event(self.current_time, EType.start, None)
        end_event = Event(self.max_simulation_time, EType.end, None)
        self.event_queue.enqueue(start_event)
        self.event_queue.enqueue(end_event)

    def addEvent(self, occurrence_time, e_type, next_event):
        new_event = Event(occurrence_time, e_type, next_event)
        self.event_queue.enqueue(new_event)

    def popEvent(self):
        return self.event_queue.dequeue()

    def runSimulation(self):
        print("Simulation running")

#TODO: schedule a debug event and print useful info about the system like event queue, current metrics, ecc or also trigger an event upon the occurrence of another event

'''
class Area:
    def __init__(self, width, height):
        self.width = width
        self.height = height
'''

class Node:
    def __init__(self, index, location, waypoint, current_speed):
        self.index = index
        self.location = location
        self.waypoint = waypoint
        self.current_speed = current_speed
        self.distance = self.distance_waypoint()
        self.last_time = 0
    
    def distance_waypoint(self):
        self.distance = math.sqrt((self.waypoint[0]-self.location[0])**2+(self.waypoint[1]-self.location[1])**2)


class Ex3Event:
    def __init__(self, e_type, n_index=None):
        self.e_type = e_type
        self.n_index = n_index

class Ex3Simulation:
    def __init__(self, max_time, n_nodes, area=(1000, 1000), v_max=(0, 10), measure=200, print_header=True):
        self.max_time = max_time
        self.n_nodes = n_nodes
        self.area = area
        self.v_max = v_max
        self.current_time = 0
        self.event_queue = []
        self.nodes = []
        self.average_speeds = []
        self.measure = measure
        self.print_header = print_header

        for i in range(n_nodes):
            position_x = random.uniform(0, area[0])
            position_y = random.uniform(0, area[1])
            waypoint_x = random.uniform(0, area[0])
            waypoint_y = random.uniform(0, area[1])
            current_speed = random.uniform(v_max[0], v_max[1])
            node = Node(i, (position_x, position_y), (waypoint_x, waypoint_y), current_speed)
            node.distance_waypoint()
            self.nodes.append(node)

        for node in self.nodes:
            if PRINT_NODES:
                print('\tTIME: {:.3f}\t| DISTANCE: {:.3f}\t| CURRENT_SPEED: {:.3f}\t| LOCATION: ({:.3f}, {:.3f}) \t| WAYPOINT: ({:.3f}, {:.3f})'.format(node.distance/node.current_speed, node.distance, node.current_speed, *node.location, *node.waypoint))
            heapq.heappush(self.event_queue, (node.distance/node.current_speed, Ex3Event(EType.waypoint_reached, node.index), node))
        

    def run(self):
        if self.print_header:
            print('\n\t--------------------------------------------------------RUNNING SIMULATION--------------------------------------------------------')
        heapq.heappush(self.event_queue, (self.measure, Ex3Event(EType.measure_speed), None))
        while self.current_time < self.max_time:
            #print('\t{:.3f}\t'.format(self.current_time), end='\t| ')
            if len(self.event_queue) > 0:
                current = heapq.heappop(self.event_queue)
                if DEBUG:
                    print(current[1].e_type.name== 'measure_speed')

                if current[1].e_type.name == 'measure_speed':
                    average = self.average_speed()
                    self.average_speeds.append(average)                    
                    self.update_event_queue(current[0])
                    self.current_time += current[0]
                    heapq.heappush(self.event_queue, (self.measure, Ex3Event(EType.measure_speed), None))
                    if DEBUG:
                        for event in self.event_queue:
                            print(event[0], event[1].e_type.name)
                    if PRINT_SPEED:
                        print('\t{:.3f}\t\t| {}\t\t| AVERAGE: {:.3f}'.format(self.current_time, current[1].e_type.name, average))
                        print('\t----------------------------------------------------------------------------------------------------------------------------------')
                else:
                    if PRINT_NODES:
                        print('\t{:.3f}\t\t| INDEX: {} \t\t| {}\t| ELAPSED: {:.3f} \t\t| DISTANCE: {:.3f}'.format(self.current_time+current[0], current[2].index, current[1].e_type.name, current[0], current[2].distance))
                    self.update_event_queue(current[0])
                    updated_node = self.update_waypoint(current[2])
                    self.current_time += current[0]
                    updated_node.last_time = self.current_time
                    
                    heapq.heappush(self.event_queue, (updated_node.distance/updated_node.current_speed, Ex3Event(EType.waypoint_reached, updated_node.index), updated_node))
                    if DEBUG:
                        for event in self.event_queue:
                            print(event[0], event[1].e_type.name)
                    if PRINT_NODES:
                        print('\tTIME: {:.3f}\t| DISTANCE: {:.3f}\t| CURRENT_SPEED: {:.3f}\t| LOCATION: ({:.3f}, {:.3f}) \t| WAYPOINT: ({:.3f}, {:.3f})'.format(updated_node.distance/updated_node.current_speed, updated_node.distance, updated_node.current_speed, *updated_node.location, *updated_node.waypoint))
                        print('\t----------------------------------------------------------------------------------------------------------------------------------')
            if len(self.average_speeds)+1 > self.max_time/self.measure:
                # this stop condition avoids to add more measure than needed
                return True

    def update_waypoint(self, node):
        position_x = node.waypoint[0]
        position_y = node.waypoint[1]
        waypoint_x = random.uniform(0, self.area[0])
        waypoint_y = random.uniform(0, self.area[1])
        current_speed = random.uniform(self.v_max[0], self.v_max[1])
        node = Node(node.index, (position_x, position_y), (waypoint_x, waypoint_y), current_speed)
        node.distance_waypoint()
        return node

    def update_event_queue(self, delta_time):
        temp_queue = []
        while self.event_queue:
            temp_item = heapq.heappop(self.event_queue)
            temp_time = temp_item[0] - delta_time
            temp_queue.append((temp_time, temp_item[1], temp_item[2]))
        
        for item in temp_queue:
            heapq.heappush(self.event_queue, item)

    def average_speed(self):
        total_speed = 0
        for node in self.event_queue:
            if node[1].e_type.name == 'waypoint_reached':
                total_speed += node[2].current_speed
        return total_speed/self.n_nodes



def exercise3_monte_carlo(sim_time, mea_time, n_nodes, v_max, n_trials=120):

    trial_results = []
    trial_avg = []
    for i in range(n_trials):
        tmp_sim = Ex3Simulation(sim_time, n_nodes, measure=mea_time, v_max=v_max, print_header=False)
        tmp_sim.run()
        trial_results.append(tmp_sim.average_speeds) #sum(tmp_sim.average_speeds)/len(tmp_sim.average_speeds)
        trial_avg.append(sum(tmp_sim.average_speeds)/len(tmp_sim.average_speeds))
        if i%10 == 0:
            print('\t[{}]\t| AVERAGE: {:.3f}'.format(i, trial_avg[i]))

    trials = np.array(trial_results)
    y_trials = []
    #print(trials.shape)
    for col in range(trials.shape[1]):
        y_trials.append(sum(trials[:, col])/trials.shape[0])

    x = range(0, sim_time, mea_time)
    plt.title('Sim={}s, n_nodes={}, measure_speed={}s, v_max={}'.format(sim_time, n_nodes, mea_time, v_max[1]))
    plt.plot(x, y_trials, linestyle='-', linewidth=1, markersize=2)
    plt.xlabel('time (s)')
    plt.ylabel('average speed (m/s)')
    plt.show()

    x_avg = range(0, n_trials, 1)
    plt.title('Sim={}s, n_nodes={}, measure_speed={}s, v_max={}'.format(sim_time, n_nodes, mea_time, v_max[1]))
    plt.plot(x_avg, trial_avg, linestyle='-', linewidth=1, markersize=2)
    plt.xlabel('trial')
    plt.ylabel('average speed (m/s)')
    plt.show()
    
            

# --------------------------------------------

# Exercise 1

simulator = Simulator()
#Point 1.4: To test the insertion of more events in the queue
simulator.addEvent(60, EType.arrival, None)
simulator.addEvent(50, EType.departure, None)
simulator.addEvent(80, EType.arrival, None)

e_pop = simulator.popEvent()

for e in simulator.event_queue:
    print(e.occurrence_time)
    print(e.e_type)
    if e.next_event is not None: print(e.next_event)
    print("-----")

print("°°°°°°°°°°°")
print(simulator.event_queue)


# --------------------------------------------

# Exercise 2


# --------------------------------------------

# Exercise 3

print("Exercise 3 \n")

print("\tIn this steps we built the simulation and validated that works; this is an example, further results are described in the report.")
sim_time = 1000
mea_time = 5
n_nodes = 50
v_max = (0, 10)
print("\n\t1. simulation_time={}\n\t2. measure_time={}\n\t3. n_nodes={}\n\t4. range(v_max)={}".format(sim_time, mea_time, n_nodes, v_max))
ex3sim = Ex3Simulation(sim_time, n_nodes, measure=mea_time, v_max=v_max)
ex3sim.run()
x = range(0, sim_time, mea_time)
plt.title('Sim={}s, n_nodes={}, measure_speed={}s, v_max={}'.format(sim_time, n_nodes, mea_time, v_max[1]))
plt.plot(x, ex3sim.average_speeds, linestyle='-', linewidth=1, markersize=2)
plt.xlabel('time (s)')
plt.ylabel('average speed (m/s)')
plt.show()

trials = 120

print("\n\n\t5. Montecarlo Simulation with {} trials and velocity range {}".format(trials, v_max))
exercise3_monte_carlo(sim_time, mea_time, n_nodes, v_max, n_trials=trials)

v_max = (1, 10)
print("\n\n\t6. Montecarlo Simulation with {} trials and velocity range {}".format(trials, v_max))
exercise3_monte_carlo(sim_time, mea_time, n_nodes, v_max=v_max, n_trials=trials)


# --------------------------------------------
