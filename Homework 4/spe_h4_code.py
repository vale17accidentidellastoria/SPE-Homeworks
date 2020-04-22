import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import statsmodels.api as sm
from scipy.spatial import distance
import enum

random.seed()


class EType(enum.Enum):
    """
    Enumerator class that defines the types of events
    """
    start = "START"
    end = "END"
    arrival = "ARRIVAL"
    departure = "DEPARTURE"
    debug = "DEBUG"


class Event:
    """
    Class that defines an event with the three attributes: occurrence time, event type, pointer to the next event
    """
    def __init__(self, occurrence_time, e_type, next_event):
        self.occurrence_time = occurrence_time
        self.e_type = e_type
        self.next_event = next_event


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
        self.total_num_packets_system = []
        self.total_temporal_values = []
        self.areas_under_qt = []
        self.waiting_times_list = []
        self.epochs_packets_served = []

    def initStartEndEventsQueue(self): # this method adds the START and END queue events to initialize the empty event queue
        start_event = Event(self.current_time, EType.start, None)
        end_event = Event(self.max_simulation_time, EType.end, None)
        self.event_queue.enqueue(start_event)
        self.event_queue.enqueue(end_event)

    def initDebugEvents(self, rate): # to add a DEBUG event at time intervals of rate=100
        for i in range(rate, self.max_simulation_time, rate):
            new_debug = Event(i, EType.debug, None)
            self.event_queue.enqueue(new_debug)

    def plotNumPacketsSystemTime(self, lambda_rate, mu_rate, avg_packets_stationary):
        x_axis = self.total_temporal_values
        y_axis = self.total_num_packets_system

        plt.bar(x_axis, y_axis, label="Instantaneous system utilisation", zorder=1)
        plt.hlines(avg_packets_stationary, min(x_axis), max(x_axis), colors="r", linestyles="dashed", label="Theoretical average", zorder=3)

        plt.plot(x_axis, self.areas_under_qt, color='orange', linewidth=4, zorder=2, label="Average system utilization")
        plt.xlabel("time")
        plt.ylabel("# packets in the system (queue + in service)")
        plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
        plt.legend(loc="upper right")

        plt.show()

    def plotAvgQueueWaitTime(self, lambda_rate, mu_rate, avg_packets_wait_queue):
        x_axis = np.arange(1, len(self.waiting_times_list) + 1)
        y_axis = []

        for packet in x_axis:
            y_axis.append(sum(self.waiting_times_list[:packet]) / packet)

        plt.hlines(avg_packets_wait_queue, min(x_axis), max(x_axis), colors="r", linestyles="dashed",
                   label="Theoretical average", zorder=3)

        plt.plot(x_axis, y_axis, color='orange', linewidth=4, zorder=2,
                 label="Average waiting time")
        plt.xlabel("# packets")
        plt.ylabel("average waiting time in queue")
        plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
        plt.legend(loc="upper right")

        plt.show()

        '''
        #plots the same as before but with temporal values in the x-axis instead of the # of packets
        x_axis = self.epochs_packets_served  # np.arange(1, len(self.waiting_times_list) + 1)
        print(self.epochs_packets_served)
        y_axis = []

        for packet in range(0, len(x_axis)):
            if packet > 0:
                y_axis.append(sum(self.waiting_times_list[:packet]) / packet)
            else:
                y_axis.append(0)

        plt.hlines(avg_packets_wait_queue, min(x_axis), max(x_axis), colors="r", linestyles="dashed",
                   label="Theoretical average", zorder=3)

        plt.plot(x_axis, y_axis, color='orange', linewidth=4, zorder=2,
                 label="Average waiting time")
        plt.xlabel("# packets")
        plt.ylabel("average waiting time in queue")
        plt.title("M/M/1, \u03BB = " + str(lambda_rate) + ", \u03BC = " + str(mu_rate))
        plt.legend(loc="upper right")

        plt.show()
        '''

    def runSimulation(self, arrival_rate_lambda, service_rate_mu):
        system = System() # create an instance of our system
        area_qt = 0

        arrivals_queue = []

        #This is our event manager loop (Ex 1.5)
        while 1:
            current_event = self.event_queue.dequeue() # get the first event in the event queue
            self.current_time = current_event.occurrence_time # get the current time for the system according to the occurrence time of the considered event
            event_type = current_event.e_type.value

            if self.current_time > 0:
                area_qt += (system.getTotalNumPacketsSystem()*(self.current_time - system.getTimeLastEvent()))
                self.areas_under_qt.append(area_qt/self.current_time)
            else:
                area_qt = 0
                self.areas_under_qt.append(area_qt)

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


# --------------------------------------------

# Exercise 1

MAX_SIMULATION_TIME = 1000
DEBUG_RATE = 100

lambdas = [0.2, 3, 10] # list of different values of arrival rates
mus = [0.4, 5, 30] # list of different values of service rate

print("Exercise 1 \n")

for arrival_rate, service_rate in zip(lambdas, mus):
    print("-- Simulation running with \u03BB = " + str(arrival_rate) + ", \u03BC = " + str(service_rate) + " --\n")
    simulator = Simulator(MAX_SIMULATION_TIME, DEBUG_RATE)

    simulator.runSimulation(arrival_rate, service_rate)

    rho = arrival_rate / service_rate
    theoretical_avg_packets_stationary = rho / (1 - rho)
    simulator.plotNumPacketsSystemTime(arrival_rate, service_rate, theoretical_avg_packets_stationary)

    print("Plotted number of packets in the system (queue + in service) against time with:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =", theoretical_avg_packets_stationary, "\n")

    theoretical_avg_packets_wait_queue = pow(rho, 2) / (arrival_rate * (1 - rho))
    simulator.plotAvgQueueWaitTime(arrival_rate, service_rate, theoretical_avg_packets_wait_queue)

    print("Plotted average waiting times for packets in queue:")
    print("\t\u03C1 = " + str(rho))
    print("\ttheoretical average number of packets in the system in stationary conditions =",
          theoretical_avg_packets_wait_queue, "\n")

    print("############\n")


# --------------------------------------------

# Exercise 2


# --------------------------------------------

# Exercise 3


# --------------------------------------------
