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
import gc
from statsmodels.distributions.empirical_distribution import ECDF

random.seed()


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
    def __init__(self, occurrence_time, e_type, next_event, server_index=None):
        self.occurrence_time = occurrence_time
        self.e_type = e_type
        self.next_event = next_event
        self.server_index = server_index


class Queue:
    """
    Class that defines some basic operations that will allow us to manipulate the event queue
    """
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def __iter__(self):
        return iter(self.queue)

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    def emptyQueue(self):
        self.queue = []

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


class System:
    """
    Class created in order to access some parameter of our system like the server status and the number of elements in the queue
    """
    def __init__(self):
        self.server_status = 0 # the server status can be either 0 (=idle) or 1 (=busy)
        self.num_packet_queue = 0
        self.time_last_event = 0

    def getStatus(self):
        return self.server_status

    def setStatus(self, value):
        self.server_status = value

    def incrementQueue(self):
        self.num_packet_queue += 1

    def decrementQueue(self):
        if self.num_packet_queue > 0:
            self.num_packet_queue -= 1

    def getNumQueue(self):
        return self.num_packet_queue

    def setNumQueue(self, value):
        self.num_packet_queue = value

    def getTotalNumPacketsSystem(self):
        return self.getNumQueue() + self.getStatus()

    def getTimeLastEvent(self):
        return self.time_last_event

    def setTimeLastEvent(self, value):
        self.time_last_event = value


class Simulator:
    """
    This class let us define all the elements required to set up and then run the simulation we want to accomplish
    """
    def __init__(self, max_time, d_rate):
        self.event_queue = Queue() # initialize the event queue
        self.current_time = 0 # set the current time to 0
        self.max_simulation_time = max_time # set a maximum simulation time
        self.initStartEndEventsQueue() # to insert the start event and end event in the queue
        self.d_rate = d_rate # the rate for intervals of debug events
        self.initDebugEvents(d_rate) # to add some debug events in order to obtain useful information about the system

        #For the number of packets in the system
        self.total_num_packets_system = [] # number of packets in the system at each time
        self.total_temporal_values = [] # to keep track of the occurrence time of each event
        self.avg_system_utilization = [] # to measure the average system utilization

        #For the average waiting time of packets in the queue
        self.waiting_times_list = [] # the waiting time for each packet
        self.epochs_packets_served = [] # the time at which each packet is served and exits from queue
        self.avg_waiting_times_list = [] # the averagewaiting time for each packet

    def initStartEndEventsQueue(self): # this method adds the START and END queue events to initialize the empty event queue
        start_event = Event(self.current_time, EType.start, None)
        end_event = Event(self.max_simulation_time, EType.end, None)
        self.event_queue.enqueue(start_event)
        self.event_queue.enqueue(end_event)

    def initDebugEvents(self, rate): # to add a DEBUG event at time intervals of rate=100
        for i in range(rate, self.max_simulation_time, rate):
            new_debug = Event(i, EType.debug, None)
            self.event_queue.enqueue(new_debug)

    def plotNumPacketsSystemTime(self, lambda_rate, mu_rate, avg_packets_stationary, epochs=0):
        x_axis = []
        y_axis = []

        if epochs == 0:
            x_axis = self.total_temporal_values
            y_axis = self.total_num_packets_system

            plt.bar(x_axis, y_axis, label="Instantaneous system utilisation", zorder=1, alpha=0.6)
            plt.hlines(avg_packets_stationary, min(x_axis), max(x_axis), colors="r", linestyles="dashed", label="Theoretical average", zorder=3, alpha=0.8)

            plt.plot(x_axis, self.avg_system_utilization, color='magenta', linewidth=2, zorder=2, label="Average system utilization")
            plt.xlabel("time")
            plt.ylabel("# packets in the system (queue + in service)")
            plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.legend(loc="upper right")

            plt.show()

        else:
            last = 0
            start = 0

            x = np.linspace(st.expon.ppf(0.01), st.expon.ppf(0.99), 100)
            plt.plot(x, st.expon.cdf(x, loc=0, scale=1 / lambda_rate), linestyle="dashed", label="Theoretical ECDF")

            max_xlim = 0

            for i in range(1, epochs + 1):
                x_axis = [x for x in self.total_temporal_values if last < x < (100*i)]
                last = i*100
                end = start + len(x_axis) - 1
                y_axis = self.avg_system_utilization[start:end+1]
                start = end + 1

                legend_label = ("Time " + str((i-1)*100) + "-" + str(i*100) + "s")
                ecdf = ECDF(y_axis)
                max_xlim = max(np.max(ecdf.x), max_xlim)
                plt.plot(ecdf.x, ecdf.y, label=legend_label)

            plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            #plt.xlim(0, 4.0)
            plt.xlim([0, max_xlim + 0.3])
            plt.ylim([0, 1.01])
            plt.legend()
            plt.show()

    def plotAvgQueueWaitTime(self, lambda_rate, mu_rate, avg_packets_wait_queue, epochs=0):
        x_axis = []
        y_axis = []

        if epochs == 0:
            x_axis = self.epochs_packets_served
            y_axis = []

            for packet in range(0, len(x_axis)):
                if packet > 0:
                    y_axis.append(sum(self.waiting_times_list[:packet]) / packet)
                else:
                    y_axis.append(0)

            self.avg_waiting_times_list = y_axis

            plt.bar(x_axis, self.waiting_times_list, label="Instantaneous waiting time", zorder=1, color="green", alpha=0.6)

            plt.hlines(avg_packets_wait_queue, min(x_axis), max(x_axis), colors="r", linestyles="dashed",
                       label="Theoretical average", zorder=3)

            plt.plot(x_axis, y_axis, color='orange', linewidth=2, zorder=2,
                     label="Average waiting time")
            plt.xlabel("# time")
            plt.ylabel("average waiting time in queue")
            plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.legend(loc="upper right")

            plt.show()

        else:
            last = 0
            start = 0

            x = np.linspace(st.expon.ppf(0.01), st.expon.ppf(0.99), 100)
            plt.plot(x, st.expon.cdf(x, loc=0, scale=1 / lambda_rate), linestyle="dashed", label="Theoretical ECDF")

            max_xlim = 0

            for i in range(1, epochs +1):
                x_axis = [x for x in self.epochs_packets_served if last < x < (100*i)]
                last = i*100
                end = start + len(x_axis) - 1
                y_axis = self.avg_waiting_times_list[start:end+1]
                start = end + 1

                legend_label = ("Time " + str((i - 1) * 100) + "-" + str(i * 100) + "s")
                ecdf = ECDF(y_axis)
                max_xlim = max(np.max(ecdf.x), max_xlim)
                plt.plot(ecdf.x, ecdf.y, label=legend_label)

            plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.xlim([0, max_xlim + 0.3])
            plt.ylim([0, 1.01])
            plt.legend()
            plt.show()

    def runSimulation(self, arrival_rate_lambda, service_rate_mu, c=1):
        if c == 1:
            system = System() # create an instance of our system
        area_qt = 0

        arrivals_queue = [] # for the purpose of computing waiting times

        #This is our event manager loop (Ex 1.5)
        while 1:
            current_event = self.event_queue.dequeue() # get the first event in the event queue
            self.current_time = current_event.occurrence_time # get the current time for the system according to the occurrence time of the considered event
            event_type = current_event.e_type.value

            if self.current_time > 0:
                area_qt += (system.getTotalNumPacketsSystem()*(self.current_time - system.getTimeLastEvent()))
                self.avg_system_utilization.append(area_qt / self.current_time)
            else:
                area_qt = 0
                self.avg_system_utilization.append(area_qt)

            current_num_packets_system = system.getTotalNumPacketsSystem()
            self.total_num_packets_system.append(current_num_packets_system)
            self.total_temporal_values.append(self.current_time)

            if event_type == "START":
                print("=> START of simulation at time t =", self.current_time, "\n")
                new_time = random.expovariate(arrival_rate_lambda)
                new_arrival = Event(new_time, EType.arrival, None)
                self.event_queue.enqueue(new_arrival)

            elif event_type == "DEBUG":
                print("DEBUG EVENT #", int(self.current_time / self.d_rate))
                print("The current time is:", self.current_time)
                if system.getStatus() == 0:
                    print("The server is IDLE")
                else:
                    print("The server is BUSY")
                print("The number of packets in the queue is", system.getNumQueue())
                print("---------")

            elif event_type == "ARRIVAL":
                #print("t = ", self.current_time, "ARRIVAL")
                if system.getStatus() == 0:
                    system.setStatus(1)
                    new_arrival = Event(self.current_time + random.expovariate(arrival_rate_lambda), EType.arrival, None)
                    new_departure = Event(self.current_time + random.expovariate(service_rate_mu), EType.departure, None)
                    self.event_queue.enqueue(new_arrival)
                    self.event_queue.enqueue(new_departure)

                    serving_time = self.current_time
                    arrival_time = self.current_time
                    waiting_time = serving_time - arrival_time
                    self.waiting_times_list.append(waiting_time)

                    self.epochs_packets_served.append(self.current_time) # if necessary

                elif system.getStatus() == 1:
                    system.incrementQueue()
                    new_arrival = Event(self.current_time + random.expovariate(arrival_rate_lambda), EType.arrival, None)
                    self.event_queue.enqueue(new_arrival)

                    arrival_time = self.current_time
                    arrivals_queue.append(arrival_time)

            elif event_type == "DEPARTURE":
                #print("t = ", self.current_time, "DEPARTURE")
                if system.getStatus() == 1:
                    if system.getNumQueue() == 0:
                        system.setStatus(0)
                        #print("#########")
                    else:
                        system.decrementQueue()
                        new_departure = Event(self.current_time + random.expovariate(service_rate_mu), EType.departure, None)
                        self.event_queue.enqueue(new_departure)

                        arrival_time = arrivals_queue.pop(0)
                        serving_time = self.current_time
                        waiting_time = serving_time - arrival_time
                        self.waiting_times_list.append(waiting_time)

                        self.epochs_packets_served.append(self.current_time) # if necessary

            elif event_type == "END":
                print("\n=> END of simulation at time t =", self.current_time, "\n")
                self.event_queue.emptyQueue() # since we arrived at the end of our simulation we empty the event queue
                system.setStatus(0)
                system.setNumQueue(0)
                return

            system.setTimeLastEvent(self.current_time)

class SystemEx2(System):
    def __init__(self, index=0, total_num_packets_system=[], total_temporal_values=[], avg_system_utilization=[]):
        super().__init__()
        self.index = index
        self.total_num_packets_system = total_num_packets_system # number of packets in the system at each time
        self.total_temporal_values = total_temporal_values # to keep track of the occurrence time of each event
        self.avg_system_utilization = avg_system_utilization # to measure the average system utilization
        self.n_packets_served = 0

class Ex2Simulator:
    """
    This class let us define all the elements required to set up and then run the simulation we want to accomplish
    """
    def __init__(self, max_time, d_rate):
        self.event_queue = Queue() # initialize the event queue
        self.current_time = 0 # set the current time to 0
        self.max_simulation_time = max_time # set a maximum simulation time
        self.initStartEndEventsQueue() # to insert the start event and end event in the queue
        self.d_rate = d_rate # the rate for intervals of debug events
        self.initDebugEvents(d_rate) # to add some debug events in order to obtain useful information about the system

        #For the number of packets in the system
        self.total_num_packets_system = [] # number of packets in the system at each time
        self.total_temporal_values = [] # to keep track of the occurrence time of each event
        self.avg_system_utilization = [] # to measure the average system utilization

        self.systems = []
        self.queue_packets = 0

        #For the average waiting time of packets in the queue
        self.waiting_times_list = [] # the waiting time for each packet
        self.epochs_packets_served = [] # the time at which each packet is served and exits from queue
        self.avg_waiting_times_list = [] # the averagewaiting time for each packet

    def initStartEndEventsQueue(self): # this method adds the START and END queue events to initialize the empty event queue
        start_event = Event(self.current_time, EType.start, None)
        end_event = Event(self.max_simulation_time, EType.end, None)
        self.event_queue.enqueue(start_event)
        self.event_queue.enqueue(end_event)


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
    
            


    def initDebugEvents(self, rate): # to add a DEBUG event at time intervals of rate=100
        for i in range(rate, self.max_simulation_time, rate):
            new_debug = Event(i, EType.debug, None)
            self.event_queue.enqueue(new_debug)

    def decrementQueue(self):
        #print('decrement')
        if self.queue_packets > 0:
            self.queue_packets -= 1

    def plotNumPacketsSystemTime(self, lambda_rate, mu_rate, avg_packets_stationary, epochs=0):
        x_axis = []
        y_axis = []

        if epochs == 0:
            x_axis = self.total_temporal_values
            y_axis = self.total_num_packets_system

            plt.bar(x_axis, y_axis, label="Instantaneous system utilisation", zorder=1, alpha=0.6)
            plt.hlines(avg_packets_stationary, min(x_axis), max(x_axis), colors="r", linestyles="dashed", label="Theoretical average", zorder=3, alpha=0.8)

            plt.plot(x_axis, self.avg_system_utilization, color='magenta', linewidth=2, zorder=2, label="Average system utilization")
            plt.xlabel("time")
            plt.ylabel("# packets in the system (queue + in service)")
            plt.title("M/M/c, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.legend(loc="upper right")

            plt.show()

        else:
            last = 0
            start = 0

            x = np.linspace(st.expon.ppf(0.01), st.expon.ppf(0.99), 100)
            plt.plot(x, st.expon.cdf(x, loc=0, scale=1 / lambda_rate), linestyle="dashed", label="Theoretical ECDF")

            max_xlim = 0

            for i in range(1, epochs + 1):
                x_axis = [x for x in self.total_temporal_values if last < x < (100*i)]
                last = i*100
                end = start + len(x_axis) - 1
                y_axis = self.avg_system_utilization[start:end+1]
                start = end + 1

                legend_label = ("Time " + str((i-1)*100) + "-" + str(i*100) + "s")
                ecdf = ECDF(y_axis)
                max_xlim = max(np.max(ecdf.x), max_xlim)
                plt.plot(ecdf.x, ecdf.y, label=legend_label)

            plt.title("M/M/c, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            #plt.xlim(0, 4.0)
            plt.xlim([0, max_xlim + 0.3])
            plt.ylim([0, 1.01])
            plt.legend()
            plt.show()

    def plotAvgQueueWaitTime(self, lambda_rate, mu_rate, avg_packets_wait_queue, epochs=0):
        x_axis = []
        y_axis = []

        if epochs == 0:
            x_axis = self.epochs_packets_served
            y_axis = []

            for packet in range(0, len(x_axis)):
                if packet > 0:
                    y_axis.append(sum(self.waiting_times_list[:packet]) / packet)
                else:
                    y_axis.append(0)

            self.avg_waiting_times_list = y_axis

            plt.bar(x_axis, self.waiting_times_list, label="Instantaneous waiting time", zorder=1, color="green", alpha=0.6)

            plt.hlines(avg_packets_wait_queue, min(x_axis), max(x_axis), colors="r", linestyles="dashed",
                       label="Theoretical average", zorder=3)

            plt.plot(x_axis, y_axis, color='orange', linewidth=2, zorder=2,
                     label="Average waiting time")
            plt.xlabel("# time")
            plt.ylabel("average waiting time in queue")
            plt.title("M/M/c, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.legend(loc="upper right")

            plt.show()

        else:
            last = 0
            start = 0

            x = np.linspace(st.expon.ppf(0.01), st.expon.ppf(0.99), 100)
            plt.plot(x, st.expon.cdf(x, loc=0, scale=1 / lambda_rate), linestyle="dashed", label="Theoretical ECDF")

            max_xlim = 0

            for i in range(1, epochs +1):
                x_axis = [x for x in self.epochs_packets_served if last < x < (100*i)]
                last = i*100
                end = start + len(x_axis) - 1
                y_axis = self.avg_waiting_times_list[start:end+1]
                start = end + 1

                legend_label = ("Time " + str((i - 1) * 100) + "-" + str(i * 100) + "s")
                ecdf = ECDF(y_axis)
                max_xlim = max(np.max(ecdf.x), max_xlim)
                plt.plot(ecdf.x, ecdf.y, label=legend_label)

            plt.title("M/M/c, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
            plt.xlim([0, max_xlim + 0.3])
            plt.ylim([0, 1.01])
            plt.legend()
            plt.show()

    def free_server_index(self):
        for system in self.systems:
            if system.getStatus() == 0:
                return system.index
        return 0

    def free_lowcount_index(self):
        lowest = None
        for system in self.systems:
            if lowest == None:
                lowest = system.n_packets_served
            elif system.n_packets_served < lowest:
                lowest = system.n_packets_served
        #print(lowest)
        for system in self.systems:
            if system.getStatus() == 0 and system.n_packets_served <= lowest:
                return system.index
        return 0

    def runSimulation(self, arrival_rate_lambda, service_rate_mu, c=1, balancer=False):
        #system
        for i in range(c):
            self.systems.append(SystemEx2(index=i))
        area_qt = 0
        #print(self.systems)
        arrivals_queue = [] # for the purpose of computing waiting times

        #This is our event manager loop (Ex 1.5)
        while 1:
            current_event = self.event_queue.dequeue() # get the first event in the event queue
            self.current_time = current_event.occurrence_time # get the current time for the system according to the occurrence time of the considered event
            event_type = current_event.e_type.value
            event_index = current_event.server_index
            #if not event_index==None:
                #print(event_index)
            #print(event_type, self.queue_packets)
            tot = 0
            for system in self.systems:
                tot += system.server_status

            if self.current_time > 0:                
                area_qt += ((self.queue_packets + tot)*(self.current_time - system.getTimeLastEvent()))
                self.avg_system_utilization.append(area_qt / self.current_time)
            else:
                area_qt = 0
                self.avg_system_utilization.append(area_qt)

            current_num_packets_system = self.queue_packets + tot

            self.total_num_packets_system.append(current_num_packets_system)
            self.total_temporal_values.append(self.current_time)

            if event_type == "START":
                print("=> START of simulation at time t =", self.current_time, "\n")
                new_time = random.expovariate(arrival_rate_lambda)
                new_arrival = Event(new_time, EType.arrival, None)
                self.event_queue.enqueue(new_arrival)

                #print(self.event_queue.queue)

            elif event_type == "DEBUG":
                print("DEBUG EVENT #", int(self.current_time / self.d_rate))
                print("The current time is:", self.current_time)
                for system in self.systems:
                    if system.getStatus() == 0:
                        print("Server {} is IDLE".format(system.index))
                    else:
                        print("Server {} is BUSY".format(system.index))
                print("The number of packets in the queue is", self.queue_packets)
                print("---------")

            elif event_type == "ARRIVAL":
                #print("t = ", self.current_time, "ARRIVAL")
                if balancer:
                    index_free_system = self.free_lowcount_index()#self.free_server_index() 
                else:
                    index_free_system = self.free_server_index()

                system = self.systems[index_free_system]
                
                if system.getStatus() == 0:
                    system.setStatus(1)
                    system.n_packets_served += 1
                    new_arrival = Event(self.current_time + random.expovariate(arrival_rate_lambda), EType.arrival, None)
                    new_departure = Event(self.current_time + random.expovariate(service_rate_mu), EType.departure, None, server_index=system.index)
                    self.event_queue.enqueue(new_arrival)
                    self.event_queue.enqueue(new_departure)

                    serving_time = self.current_time
                    arrival_time = self.current_time
                    waiting_time = serving_time - arrival_time
                    self.waiting_times_list.append(waiting_time)

                    self.epochs_packets_served.append(self.current_time) # if necessary

                elif system.getStatus() == 1:
                    self.queue_packets += 1
                    new_arrival = Event(self.current_time + random.expovariate(arrival_rate_lambda), EType.arrival, None)
                    self.event_queue.enqueue(new_arrival)

                    arrival_time = self.current_time
                    arrivals_queue.append(arrival_time)

                #print('break worked after', i)

            elif event_type == "DEPARTURE":
                system = self.systems[event_index]

                if system.getStatus() == 1:
                    if self.queue_packets == 0:
                        system.setStatus(0)
                        #print("###SET STATUS 0######")
                    else:
                        if self.queue_packets > 0:
                            self.queue_packets -= 1

                        new_departure = Event(self.current_time + random.expovariate(service_rate_mu), EType.departure, None, server_index=event_index)
                        self.event_queue.enqueue(new_departure)

                        arrival_time = arrivals_queue.pop(0)
                        serving_time = self.current_time
                        waiting_time = serving_time - arrival_time
                        self.waiting_times_list.append(waiting_time)

                        self.epochs_packets_served.append(self.current_time) # if necessary

            elif event_type == "END":
                print("\n=> END of simulation at time t =", self.current_time, "\n")
                self.event_queue.emptyQueue() # since we arrived at the end of our simulation we empty the event queue
                for system in self.systems:
                    system.setStatus(0)
                    self.queue_packets = 0
                return
            
            for system in self.systems:
                system.setTimeLastEvent(self.current_time)

class MMcQueue(object):
    def __init__(self, arrival, departure, capacity):
        self._arrival = float(arrival)
        self._departure = float(departure)
        self._capacity = capacity
        self._rou = self._arrival / self._departure / self._capacity

        # init the parameter as if the capacity == 0
        powerTerm = 1.0
        factorTerm = 1.0
        preSum = 1.0
        # Loop through `1` to `self._capacity` to get each term and preSum
        for i in range(1, self._capacity + 1):
            powerTerm *= self._arrival / self._departure
            factorTerm /= i
            preSum += powerTerm * factorTerm
        self._finalTerm = powerTerm * factorTerm
        preSum -= self._finalTerm
        self._p0 = 1.0 / (preSum + self._finalTerm / (1 - self._rou))
        self._pc = self._finalTerm * self._p0
        self._probSum = preSum * self._p0


    @property
    def arrival(self):
        return self._arrival

    @property
    def departure(self):
        return self._departure

    @property
    def capacity(self):
        return self._capacity

    def getPk(self, k):
        """
        Return the probability when there are `k` packets in the system
        """
        if k == 0:
            return self._p0
        elif k == self._capacity:
            return self._pc
        elif k < self._capacity:
            factorTerm = 1.0 / math.factorial(k)
            powerTerm = math.pow(self._arrival / self._departure, k)
            return self._p0 * factorTerm * powerTerm
        else:
            return self._finalTerm * math.pow(self._rou, k - self._capacity) * self._p0

    def getQueueProb(self):
        """
        Return the probability when a packet comes, it needs to queue in the buffer.
        That is, P(W>0) = 1 - P(N < c)
        Also known as Erlang-C function
        """
        return 1.0 - self._probSum

    def getIdleProb(self):
        """
        Return the probability when the sever is idle.
        That is , P(N=0)
        """
        return self._p0

    def getAvgPackets(self):
        """
        Return the average number of packets in the system (in service and in the queue)
        """
        return self._rou / (1 - self._rou) * self.getQueueProb() + self._capacity * self._rou

    def getAvgQueueTime(self):
        """
        Return the average time of packets spending in the queue
        """
        return self.getQueueProb() / (self._capacity * self._departure - self._arrival)

    def getAvgQueuePacket_Given(self):
        """
        Given there is packet in the queue,
        return the average number of packets in the queue
        """
        return self._finalTerm * self._p0 / (1.0 - self._rou) / (1.0 - self._rou)

    def getAvgQueueTime_Given(self):
        """
        Given a packet must wait, 
        return the average time of this packet spending in the queue
        """
        if self.getQueueProb() == 0:
            return 0
        return self.getAvgQueuePacket_Given() / (self.getQueueProb() * self._arrival)

    def getAvgResponseTime(self):
        """
        Return the average time of packets spending in the system (in service and in the queue)
        """
        return self.getAvgQueueTime() + 1.0 / self._departure

    def getAvgPacketInSystem(self):
        """
        Return the average number of packets in the system.
        """
        return self.getAvgResponseTime() * self._arrival

    def getAvgBusyServer(self):
        """
        Return the average number of busy Server.
        """
        return self.arrival / self.departure


    def getPorbWhenQueueTimeLargerThan(self, queueTime):
        """
        Return the probability when the queuing time of the packet is larger than `queueTime`
        That is P(W > queueTime) = 1 - P(W <= queueTime)
        """
        firstTerm = self._pc / (1.0 - self._rou)
        expTerm = - self._capacity * self._departure * (1.0 - self._rou) * queueTime
        secondTerm = math.exp(expTerm)
        return firstTerm * secondTerm
# --------------------------------------------

# Exercise 1

MAX_SIMULATION_TIME = 1000
DEBUG_RATE = 100

#lambdas = [0.2, 3, 10] # list of different values of arrival rates
#mus = [0.4, 5, 30] # list of different values of service rate
lambdas = [10]
mus = [15]

print("Exercise 1 \n")

for arrival_rate, service_rate in zip(lambdas, mus):
    print("-- Simulation running with \u03BB = " + str(arrival_rate) + ", \u03BC = " + str(service_rate) + " --\n")
    simulator = Simulator(MAX_SIMULATION_TIME, DEBUG_RATE)

    simulator.runSimulation(arrival_rate, service_rate, c=1)

    rho = arrival_rate / service_rate
    theoretical_avg_packets_stationary = rho / (1 - rho)
    simulator.plotNumPacketsSystemTime(arrival_rate, service_rate, theoretical_avg_packets_stationary, epochs=0)

    print("Plotted number of packets in the system (queue + in service) against time with:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =", theoretical_avg_packets_stationary, "\n")

    theoretical_avg_packets_wait_queue = pow(rho, 2) / (arrival_rate * (1 - rho))
    simulator.plotAvgQueueWaitTime(arrival_rate, service_rate, theoretical_avg_packets_wait_queue, epochs=0)

    print("Plotted average waiting times for packets in queue:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =",
          theoretical_avg_packets_wait_queue, "\n")

    simulator.plotNumPacketsSystemTime(arrival_rate, service_rate, theoretical_avg_packets_stationary, epochs=10)
    print("Plotted ECDF of number of packets in the system for 10 different epochs!")
    simulator.plotAvgQueueWaitTime(arrival_rate, service_rate, theoretical_avg_packets_wait_queue, epochs=10)
    print("Plotted ECDF of queue waiting time for 10 different epochs!")

    print("############\n")


# --------------------------------------------
gc.collect()
# Exercise 2

print("Exercise 2 \n")

MAX_SIMULATION_TIME = 1000
DEBUG_RATE = 100

#lambdas = [0.2, 3, 10] # list of different values of arrival rates
#mus = [0.4, 5, 30] # list of different values of service rate
lambdas = [10]
mus = [5]
servers = 5

for arrival_rate, service_rate in zip(lambdas, mus):
    print("-- Simulation running with \u03BB = " + str(arrival_rate) + ", \u03BC = " + str(service_rate) + " --\n")
    simulator = Ex2Simulator(MAX_SIMULATION_TIME, DEBUG_RATE)

    simulator.runSimulation(arrival_rate, service_rate, c=servers)

    queue = MMcQueue(arrival_rate, service_rate, servers)
    theoretical_avg_packets_stationary = queue.getAvgPacketInSystem()

    rho = arrival_rate / (servers*service_rate)
    simulator.plotNumPacketsSystemTime(arrival_rate, service_rate, theoretical_avg_packets_stationary, epochs=0)

    print("Plotted number of packets in the system (queue + in service) against time with:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =", theoretical_avg_packets_stationary, "\n")

    theoretical_avg_packets_wait_queue = queue.getAvgQueueTime()#pow(rho, 2) / (arrival_rate * (1 - rho))
    simulator.plotAvgQueueWaitTime(arrival_rate, service_rate, theoretical_avg_packets_wait_queue, epochs=0)

    print("Plotted average waiting times for packets in queue:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =",
          theoretical_avg_packets_wait_queue, "\n")

    simulator.plotNumPacketsSystemTime(arrival_rate, service_rate, theoretical_avg_packets_stationary, epochs=10)
    print("Plotted ECDF of number of packets in the system for 10 different epochs!")

    simulator.plotAvgQueueWaitTime(arrival_rate, service_rate, theoretical_avg_packets_wait_queue, epochs=10)
    print("Plotted ECDF of queue waiting time for 10 different epochs!")
    
    server_served_y = []
    for system in simulator.systems:
        server_served_y.append(system.n_packets_served)
    servers_id = np.arange(servers)

    plt.title('Packets served without balancer')
    plt.bar(servers_id, server_served_y, align='center', alpha=0.5)
    plt.xticks(servers_id, servers_id)
    plt.xlabel('server id')
    plt.ylabel('Served Packets')

    plt.show()
    print("Plotted Histogram showing the number of packets served by each server where c={}.".format(servers))

    simulator_balancer = Ex2Simulator(MAX_SIMULATION_TIME, DEBUG_RATE)

    simulator_balancer.runSimulation(arrival_rate, service_rate, c=servers, balancer=True)

    server_served_y = []
    for system in simulator_balancer.systems:
        server_served_y.append(system.n_packets_served)
    servers_id = np.arange(servers)

    plt.title('Packets served with balancer')
    plt.bar(servers_id, server_served_y, align='center', alpha=0.5)
    plt.xticks(servers_id, servers_id)
    plt.xlabel('server id')
    plt.ylabel('Served Packets')
    print("Plotted Histogram showing the number of packets served by each server where c={} with balancer.".format(servers))

    plt.show()

    print("############\n")
    

# --------------------------------------------
gc.collect()
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
