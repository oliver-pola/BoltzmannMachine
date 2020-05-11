#!/usr/bin/env python

from enum import Enum
import numpy as np


Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')

class Step:
    def __init__(self, temperature, epochs):
        self.temperature = temperature
        self.epochs = epochs

numInputUnits = 4
numOutputUnits = 4
numHiddenUnits = 2

numVisibleUnits = numInputUnits + numOutputUnits
numUnits = numVisibleUnits+numHiddenUnits

annealingSchedule = [Step(20.,2),
                     Step(15.,2),
                     Step(12.,2),
                     Step(10.,4)]

coocurranceCycle = Step(10.,10)

weights = np.zeros((numUnits,numUnits))
states = np.zeros(numUnits)
energy = np.zeros(numUnits)

connections = np.zeros((numUnits,numUnits), dtype=np.int)
for i in range(numInputUnits):
    # connection inside input layer
    for j in range(i+1,numInputUnits):
        connections[i,j] = 1
    # connections input to output layer
    for j in range(1,numHiddenUnits+1):
        connections[i,-j] = 1   

            
for i in range(numOutputUnits):
    # connections inside output layer
    for j in range(i+1,numOutputUnits):
        connections[i+numInputUnits,j+numInputUnits] = 1
    # connections output to hidden layer
    for j in range(1,numHiddenUnits+1):
        connections[i+numInputUnits,-j] = 1  

for i in range(numHiddenUnits,0,-1):
    # connections inside hidden layer
    for j in range(i-1,0,-1):
        connections[-i,-j] = 1

        
# Get matrix of indices from connections that are non zero
valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections[valid] = np.arange(1,numConnections+1)
# Mirror the connection matrix to also fill the lower left half
# Also gives existing connection pair the same values so they can be identified
# All other matrix fields get -1 as value
connections = connections + connections.T - 1

def propagate(temperature, clamp):
    global energy, states, weights
    
    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = numUnits
    else:
        numUnitsToSelect = numHiddenUnits + numOutputUnits

    for i in range(numUnitsToSelect):
        # Calculating the energy of a randomly selected unit    
        unit=numUnits-np.random.randint(1,numUnitsToSelect+1)
        energy[unit] = np.dot(weights[unit,:], states)
        
        p = 1. / (1.+ np.exp(-energy[unit] / temperature))
        states[unit] = 1. if  np.random.uniform() <= p else 0 
                    
def anneal(annealingSchedule, clamp):
    for step in annealingSchedule:
        for epoch in range(step.epochs):
            propagate(step.temperature, clamp)
    
# Counts how many active connections each unit has
def sumCoocurrance(clamp):                        
    sums = np.zeros(numConnections)
    for epoch in range(coocurranceCycle.epochs):
        propagate(coocurranceCycle.temperature, clamp)
        for i in range(numUnits):
            if(states[i] == 1):
                for j in range(i+1,numUnits):
                    if(connections[i,j]>-1 and states[j] ==1):
                        sums[connections[i,j]] += 1   
    return sums
     
def updateWeights(pplus, pminus):
    global weights
    for i in range(numUnits):
        for j in range(i+1,numUnits):            
            if connections[i,j] > -1:
                index = connections[i,j]
                # Add/Subtract 2 to the weight of an active connections depending
                # on if there where more active connections in the positive or negative phase
                weights[i,j] += 2*np.sign(pplus[index] - pminus[index])
                weights[j,i] = weights[i,j]

def recall(pattern):
    global states
        
    # Setting pattern to recall
    states[0:numInputUnits] = pattern
     
    # Assigning random values to the hidden and output states
    states[-(numHiddenUnits+numOutputUnits):] = np.random.choice([0,1],numHiddenUnits+numOutputUnits)
    anneal(annealingSchedule, Clamp.INPUT_UNITS)
    
    return states[numInputUnits:numInputUnits+numOutputUnits]
    
def addNoise(pattern):
    probabilities = 0.8*pattern+0.05
    uniform = np.random.random(numVisibleUnits)    
    return (uniform < probabilities).astype(int)
    
    
def learn(patterns):
    global states, weights

    numPatterns = patterns.shape[0]    
    trials=numPatterns*coocurranceCycle.epochs
    weights = np.zeros((numUnits,numUnits))
            
    for i in range(1800):
        # Positive phase
        pplus = np.zeros(numConnections)
        for pattern in patterns:
            
            # Setting visible units values (inputs and outputs)
            states[0:numVisibleUnits] = addNoise(pattern)
    
            # Assigning random values to the hidden units
            states[-numHiddenUnits:] = np.random.choice([0,1],numHiddenUnits)
    
            anneal(annealingSchedule, Clamp.VISIBLE_UNITS)
            pplus += sumCoocurrance(Clamp.VISIBLE_UNITS)
        pplus /= trials
        
        # Negative phase
        states = np.random.choice([0,1],numUnits)          
        anneal(annealingSchedule, Clamp.NONE)
        pminus = sumCoocurrance(Clamp.NONE) / coocurranceCycle.epochs
       
        updateWeights(pplus,pminus)


# Encoder problem input/output pairs that will be learned
patterns = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1]])
learn(patterns)
print (weights)

print (recall(np.array([1, 0, 0, 0])))
