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

MAX_SIMULATION_TIME = 1000

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


# --------------------------------------------
