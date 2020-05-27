#!/usr/bin/env python

import numpy as np
import sys


class Boltzmann:

    class Step:
        def __init__(self, temperature, epochs):
            self.temperature = temperature
            self.epochs = epochs


    def __init__(self, visible, hidden, output, annealing, coocurance, synchron_update=False):
        """
        Initialize a Boltzmann machine with the given parameters

        visible: Number of visible units
        hidden: Number of hidden units
        output: Number of output units. If this number is not equal to visible
              a Boltzmann machine with two visible layers is created, where
              the two visible layer do not have direct connections between each other
        annealing: An annealing schedule consisting of a list of tuples in the form
              of (<temperature>, <epochs>)
        coocurance: Coocurance cycle. One tuple in the form of (<temperature>, <epochs>)
        """
        self.num_visible_units = visible
        self.num_hidden_units = hidden
        self.num_units = visible + hidden
        self.num_input_units = visible - output
        self.num_output_units = output
        self.annealing_schedule = []
        for tuple in annealing:
            self.annealing_schedule.append(self.Step(tuple[0], tuple[1]))
        self.coocurrance_cycle = self.Step(coocurance[0], coocurance[1])

        self.synchron_update = synchron_update

        self.weights = np.zeros((self.num_units, self.num_units))
        self.states = np.zeros(self.num_units)
        self.energy = np.zeros(self.num_units, dtype=np.float64)

        self.init_connections()


    def init_connections(self):
        """
        Initialize the connections matrix.
        This creates a connection matrix with field that contain numeric id for
        each connection pair.
        """
        n_in = self.num_input_units
        n_out = self.num_output_units
        n_hidden = self.num_hidden_units
        n_units = self.num_units

        self.connections = np.zeros((n_units, n_units), dtype=np.int)

        # Connections inside input layer
        for i in range(n_in):
            self.connections[i, i+1:n_units] = 1
        # Connections between input/hidden layer
        self.connections[:n_in, -n_hidden:] = 1

        for i in range(n_out):
            # Connections inside output layer
            self.connections[n_in+i, n_in+i+1:n_in+n_out] = 1
            # Connections between output/hidden layer
            self.connections[n_in+i, -n_hidden:] = 1

        # Connections inside hidden layer
        for i in range(n_hidden,1,-1):
            self.connections[-i, -i+1:] = 1

        # Get matrix of indices from connections that are non zero
        valid = np.nonzero(self.connections)
        self.num_connections = np.size(valid[0])
        # Give each connection a numerical id from 1 to num_connections
        self.connections[valid] = np.arange(1,self.num_connections+1)
        # Mirror the connection matrix to also fill the lower left half
        # Also gives existing connection pairs the same id from 0 to num_connections-1
        # All other matrix fields get -1 as value
        self.connections = self.connections + self.connections.T - 1


    def learn(self, patterns, iterations, noise_probability=0.8, noise_bias=0.05):
        patterns = np.array(patterns)
        num_patterns = patterns.shape[0]
        trials = self.coocurrance_cycle.epochs * num_patterns
        self.weights = np.zeros((self.num_units, self.num_units))

        if (patterns.shape[1] != self.num_visible_units):
            print("Error: The given learning patterns are of the wrong size")
            print(f'Expected shape (*, {self.num_visible_units}), got shape {patterns.shape}')
            sys.exit()

        visible_ones = np.ones(self.num_visible_units)
        visible_zeros = np.zeros(self.num_visible_units)
        for i in range(iterations):
            # Positive phase
            pplus = np.zeros(self.num_connections)

            for p in range(num_patterns):
                sys.stdout.write(f'epoch {i+1}/{iterations}, pattern {p+1}/{num_patterns}          \r')
                sys.stdout.flush()
                # Setting visible units values
                self.states[0:self.num_visible_units] = self.add_noise(patterns[p], noise_probability, noise_bias)

                # Give random values to hidden units
                self.states[-self.num_hidden_units:] = np.random.choice([0,1],self.num_hidden_units)

                self.anneal(self.annealing_schedule, visible_ones)
                pplus += self.sum_coocurrance(visible_ones)

            pplus/= trials

            # Negative phase
            self.states = np.random.choice([0,1], self.num_units)
            self.anneal(self.annealing_schedule, visible_zeros)
            pminus = self.sum_coocurrance(visible_zeros) / self.coocurrance_cycle.epochs

            self.update_weights(pplus, pminus)


    def recall(self, pattern, clamp_mask, output_mask=[]):
        # Check if the given pattern hast as many values as clamp_mask has clamped units
        pattern = np.array(pattern)
        if(pattern.shape[0] != np.nonzero(clamp_mask)[0].shape[0]):
            print("Error: number of given values for recall pattern does not match number of clamped units in given clamp_mask. Exiting.")
            print(f'Got pattern shape {pattern.shape}, clamped units ({np.nonzero(clamp_mask)[0].shape[0]})')
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
        if len(output_mask) == 0:
            # Return all visible units by default
            return self.states[0:self.num_visible_units]
        else:
            # Only return states that are indicated by the output_mask
            output_mask = np.array(output_mask)
            return self.states[np.where(output_mask == 1)]


    def add_noise(self, pattern, noise_probability, noise_bias):
        probabilities = noise_probability * pattern + noise_bias
        uniform = np.random.random(self.num_visible_units)
        return (uniform < probabilities).astype(int)


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
        # upper right trangle, without diagonal
        triu = np.triu_indices(self.num_units, 1)
        for epoch in range(self.coocurrance_cycle.epochs):
            self.propagate(self.coocurrance_cycle.temperature, clamp_mask)
            # keep only idx where state of that row == 1
            rows_keep = self.states[triu[0]]==1
            idx = (triu[0][rows_keep], triu[1][rows_keep])
            # keep only idx where state of that col == 1
            cols_keep = self.states[idx[1]]==1
            idx = (idx[0][cols_keep], idx[1][cols_keep])
            # apply those filters
            conn = self.connections[idx]
            # update sum +1 where conn > -1
            sums[conn[conn > -1]] += 1
        return sums


    def update_weights(self, pplus, pminus):
        # upper right trangle, without diagonal
        triu = np.triu_indices(self.num_units, 1)
        conn = self.connections[triu]
        filter = conn > -1
        idx = (triu[0][filter], triu[1][filter])
        self.weights[idx] += 2*np.sign(pplus[conn[filter]] - pminus[conn[filter]])
        # make symmetric, whole wights again
        # cant use np.tril_indices here, would have wrong order
        anti_triu = (triu[1], triu[0])
        self.weights[anti_triu] = self.weights[triu]


def Boltzmann_test():
    patterns = [[1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1]]

    b = Boltzmann(8,2,4,[(20.,2),(15.,2),(12.,2),(10.,4)], (10.,10), synchron_update=False)
    b.learn(patterns, 1800)

    print(b.weights)

    clamp_mask = np.append(np.ones(4), np.zeros(4))
    output_mask = np.append(np.zeros(4), np.ones(4))
    print(b.recall([1,0,0,0], clamp_mask, output_mask))


if __name__ == '__main__':
    Boltzmann_test()
