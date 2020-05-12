#!/usr/bin/env python

import numpy as np


class Boltzman:

    class Step:
        def __init__(self, temperature, epochs):
            self.temperature = temperature
            self.epochs = epochs

    num_units = 0;
    num_visible_units = 0
    num_hidden_units = 0
    num_input_units = 0
    num_output_units = 0
    num_connections = 0

    annealing_schedule = []
    coocurrance_cycle = None

    weights = None
    states = None
    energy = None
    connections = None

    #
    ## Initialue a Boltzman machine with the given parameters
    ## 
    ## visible: Number of visible units
    ## hidden: Number of hidden units
    ## output: Number of output units. If this number is not equal to visible
    ##          a boltzman machine with two visible layers is created, where
    ##          the two visible layer do not have direct connections between each other
    ## annealing: An annealing schedule consisting of a list of tuples in the form
    ##          of (<temperature>, <epochs>)
    ## coocurance: Coocurance cycle. One tuple in the form of (<temperature>, <epochs>)
    #
    def __init__(self, visible, hidden, output, annealing, coocurance):
        self.num_visible_units = visible
        self.num_hidden_units = hidden
        self.num_units = visible + hidden
        self.num_input_units = visible - output
        self.num_output_units = output
        for tuple in annealing:
            self.annealing_schedule.append(self.Step(tuple[0], tuple[1]))
        self.coocurrance_cycle = self.Step(coocurance[0], coocurance[1])

        self.weights = np.zeros((self.num_units, self.num_units))
        self.states = np.zeros(self.num_units)
        self.energy = np.zeros(self.num_units)

        self.init_connections()

    #
    ## Initialize the connections matrix.
    ## This creates a connection matrix with field that contain numeric id for 
    ## each connection pair.
    #
    def init_connections(self):
        self.connections = np.zeros((self.num_units, self.num_units), dtype=np.int)

        for i in range(self.num_input_units):
            # Uonnections inside input layer
            for j in range(i+1, self.num_input_units):
                self.connections[i,j] = 1
            # Uonnections between input/hidden layer
            for j in range(1,self.num_hidden_units+1):
                self.connections[i,-j] = 1

        for i in range(self.num_output_units):
            # Uonnections inside output layer
            for j in range(i+1, self.num_output_units):
                self.connections[self.num_input_units+i, self.num_input_units+j] = 1
            # Connections between output/hidden layer
            for j in range(1, self.num_hidden_units+1):
                self.connections[self.num_input_units+i,-j] = 1

        for i in range(self.num_hidden_units,0,-1):
            # Connections inside hidden layer
            for j in range(i-1,0,-1):
                self.connections [-i,-j] = 1
        
        # Get matrix of indices from connections that are non zero
        valid = np.nonzero(self.connections)
        self.num_connections = np.size(valid[0])
        # Give each connection a numerical id from 1 to num_connections
        self.connections[valid] = np.arange(1,self.num_connections+1)
        # Mirror the connection matrix to also fill the lower left half
        # Also gives existing connection pairs the same id from 0 to num_connections-1
        # All other matrix fields get -1 as value
        self.connections = self.connections + self.connections.T - 1


    #
    ##
    ##
    #
    def learn(self, patterns, iterations, noise_probability=0.8, noise_bias=0.05):
        patterns = np.array(patterns)
        num_patterns = patterns.shape[0]
        trials = self.coocurrance_cycle.epochs * num_patterns
        self.weights = np.zeros((self.num_units, self.num_units))

        if (patterns.shape[1] != self.num_visible_units):
            print("Error: The given learning patterns are of the wrong size")
            sys.exit()

        for i in range(iterations):
            # Positive phase
            pplus = np.zeros(self.num_connections)

            for pattern in patterns:
                # Setting visible units values 
                self.states[0:self.num_visible_units] = self.add_noise(pattern, noise_probability, noise_bias)

                # Give random values to hidden units
                self.states[-self.num_hidden_units:] = np.random.choice([0,1],self.num_hidden_units)

                self.anneal(self.annealing_schedule, np.ones(self.num_visible_units))
                pplus += self.sum_coocurrance(np.ones(self.num_visible_units))
            
            pplus/= trials

            # Negative phase
            self.states = np.random.choice([0,1], self.num_units)
            self.anneal(self.annealing_schedule, np.zeros(self.num_visible_units))
            pminus = self.sum_coocurrance(np.zeros(self.num_visible_units))

            self.update_weights(pplus, pminus)


    # TODO Work in progress
    def recall(self, pattern, clamp_mask):
        pattern = np.array(pattern)
        # Setting pattern to recall
        # TODO check wether has to be num_input/output_units for other cases
        self.states[0:self.num_input_units] = pattern

        # Assigning random values to the hidden and output states
        # TODO fix this for other input/output configurations/clamping masks
        self.states[-(self.num_hidden_units+self.num_output_units):] = \
                np.random.choice([0,1], self.num_hidden_units+self.num_output_units)
        self.anneal(self.annealing_schedule, clamp_mask)

        return self.states[self.num_input_units:self.num_input_units+self.num_output_units]

    
    #
    ##
    ##
    #
    def add_noise(self, pattern, noise_probability, noise_bias):
        probabilities = noise_probability * pattern + noise_bias
        uniform = np.random.random(self.num_visible_units)
        return (uniform < probabilities).astype(int)


    #
    ##
    ##
    #
    def anneal(self, schedule, clamp_mask):
        for step in schedule:
            for epoch in range(step.epochs):
                self.propagate(step.temperature, clamp_mask)

    def propagate(self, temperature, clamp_mask):
        clamp_mask = np.array(clamp_mask, dtype=np.int)
        clamp_mask = np.append(clamp_mask, np.zeros(self.num_hidden_units, dtype=np.int))

        # TODO check if this is actually the correct behaviour
        for i in np.where(clamp_mask == 0)[0]:
            # Calculating the energy of a randomly selected unit    
            unit = np.random.choice(np.where(clamp_mask == 0)[0])
            self.energy[unit] = np.dot(self.weights[unit,:], self.states)

            p = 1. / (1. + np.exp(-self.energy[unit] / temperature))
            self.states[unit] = 1. if np.random.uniform() <= p else 0

    def sum_coocurrance(self, clamp_mask):
        sums = np.zeros(self.num_connections)
        for epoch in range(self.coocurrance_cycle.epochs):
            self.propagate(self.coocurrance_cycle.temperature, clamp_mask)
            for i in range(self.num_units):
                if(self.states[i] == 1):
                    for j in range(i+1, self.num_units):
                        if(self.connections[i,j] > -1 and self.states[j] == 1):
                            sums[self.connections[i,j]] += 1

        return sums

    def update_weights(self, pplus, pminus):
        for i in range(self.num_units):
            for j in range(i+1, self.num_units):
                if (self.connections[i,j] > -1):
                    index = self.connections[i,j]
                    # Add/Subtract 2 to the weight of an active connections depending
                    # on if there where more active connections in the positive or negative phase
                    self.weights[i,j] += 2*np.sign(pplus[index] - pminus[index])
                    self.weights[j,i] = self.weights[i,j]




patterns = [[1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1]]

b = Boltzman(8,2,4,[(20.,2),(15.,2),(12.,2),(10.,4)], (10.,10))
b.learn(patterns, 1800)

print(b.weights)

clamp_mask = np.append(np.ones(4), np.zeros(4))
print(b.recall([1,0,0,0], clamp_mask))
        


