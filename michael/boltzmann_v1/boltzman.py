#!/usr/bin/env python

import numpy as np
import sys

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

    synchron_update = False

    weights = None
    states = None
    energy = None
    connections = None

    #
    ## Initialue a Boltzman machine with the given parameters
    ## 
    ## visible: Number of visible units
    ## hidden: Number of hidden units
    ## output: Number of output units. If this number is not equal to visible or 0
    ##          a boltzman machine with two visible layers is created, where
    ##          the two visible layer do not have direct connections between each other
    ## annealing: An annealing schedule consisting of a list of tuples in the form
    ##          of (<temperature>, <epochs>)
    ## coocurance: Coocurance cycle. One tuple in the form of (<temperature>, <epochs>)
    #
    def __init__(self, visible, hidden, output, annealing, coocurance, synchron_update=False):
        self.num_visible_units = visible
        self.num_hidden_units = hidden
        self.num_units = visible + hidden
        self.num_input_units = visible - output
        self.num_output_units = output
        for tuple in annealing:
            self.annealing_schedule.append(self.Step(tuple[0], tuple[1]))
        self.coocurrance_cycle = self.Step(coocurance[0], coocurance[1])

        self.synchron_update = synchron_update

        self.weights = np.zeros((self.num_units, self.num_units))
        self.states = np.zeros(self.num_units)
        self.energy = np.zeros(self.num_units, dtype=np.float128)

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
            pminus = self.sum_coocurrance(np.zeros(self.num_visible_units)) / self.coocurrance_cycle.epochs

            self.update_weights(pplus, pminus)


    #
    ##
    ##
    #
    def recall(self, pattern, clamp_mask, output_mask=[]):
        # Check if the given pattern hast as many values as clamp_mask has clamped units
        pattern = np.array(pattern)
        if(pattern.shape[0] != np.nonzero(clamp_mask)[0].shape[0]):
            print("Error: number of given values for recall pattern does not match number of clamped units in given clamp_mask. Exiting.")
            sys.exit(1)

        # Set the given clamped units states
        clamped_idxs = np.where(clamp_mask == 1)[0]
        for i in range(clamped_idxs.shape[0]):
            self.states[clamped_idxs[i]] = pattern[i]

        # Set unclamped units states to random 0,1 values
        unclamped_idxs = np.append(np.where(clamp_mask == 0)[0], \
                np.arange(self.num_visible_units, self.num_units))
        self.states[unclamped_idxs] = np.random.choice([0,1], unclamped_idxs.shape[0])

        self.anneal(self.annealing_schedule, clamp_mask)

        # Decide which states will be output based on the output_mask
        if output_mask == []:
            # Return all visible units by default
            return self.states[0:self.num_visible_units]
        else:
            # Only return states that are indicated by the output_mask
            output_mask = np.array(output_mask)
            return self.states[np.where(output_mask == 1)]

    
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
        unclamped_idxs = np.where(clamp_mask == 0)[0]

        if (self.synchron_update == True):
            # TODO Some error in here? getting overflow in np.exp Warning
            # might be depending on parameters. synchronous update might need different
            # runtime parameters to work here. It works with energy being dtype np.float128
            # But results using this update function are not good
            self.energy[unclamped_idxs] = np.dot(self.weights[unclamped_idxs,:], self.states)
            p = np.zeros(unclamped_idxs.shape[0])
            p = 1. / (1. + np.exp(-self.energy[unclamped_idxs] / temperature))

            self.states[unclamped_idxs] = np.random.uniform(size=unclamped_idxs.shape[0]) <= p
        else:
            choices = np.random.choice(unclamped_idxs, size=unclamped_idxs.shape[0])
            for i in range(unclamped_idxs.shape[0]):
                # Calculating the energy of a randomly selected unit    
                unit = choices[i]
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
