#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import sys
import time


# tf.compat.v1.disable_eager_execution()


def tf_random_choice(a, axis=0, size=None):
    # https://github.com/tensorflow/tensorflow/issues/8496
    if size is None:
        size = (1,)
    dim = a.shape[axis]
    choice_indices = tf.random.uniform(size, minval=0, maxval=dim, dtype=tf.int32)
    samples = tf.gather(a, choice_indices, axis=axis)
    return samples


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

        self.weights = tf.Variable(tf.zeros((self.num_units, self.num_units), dtype=tf.float64))
        self.states = tf.Variable(tf.zeros(self.num_units, dtype=tf.float64))
        self.energy = tf.Variable(tf.zeros(self.num_units, dtype=tf.float64))

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

        connections = np.zeros((n_units, n_units), dtype=np.int)

        # Connections inside input layer
        for i in range(n_in):
            connections[i, i+1:n_units] = 1
        # Connections between input/hidden layer
        connections[:n_in, -n_hidden:] = 1

        for i in range(n_out):
            # Connections inside output layer
            connections[n_in+i, n_in+i+1:n_in+n_out] = 1
            # Connections between output/hidden layer
            connections[n_in+i, -n_hidden:] = 1

        # Connections inside hidden layer
        for i in range(n_hidden,1,-1):
            connections[-i, -i+1:] = 1

        # Get matrix of indices from connections that are non zero
        valid = np.nonzero(connections)
        self.num_connections = np.size(valid[0])
        # Give each connection a numerical id from 1 to num_connections
        connections[valid] = np.arange(1,self.num_connections+1)
        # Mirror the connection matrix to also fill the lower left half
        # Also gives existing connection pairs the same id from 0 to num_connections-1
        # All other matrix fields get -1 as value
        connections = connections + connections.T - 1

        self.connections = tf.Variable(connections, dtype=tf.int8)


    def learn(self, patterns, iterations, noise_probability=0.8, noise_bias=0.05):
        patterns = np.array(patterns)
        num_patterns = patterns.shape[0]
        trials = self.coocurrance_cycle.epochs * num_patterns
        self.weights = tf.zeros((self.num_units, self.num_units), dtype=tf.float64)

        if (patterns.shape[1] != self.num_visible_units):
            print("Error: The given learning patterns are of the wrong size")
            print(f'Expected shape (*, {self.num_visible_units}), got shape {patterns.shape}')
            sys.exit()

        visible_ones = tf.ones(self.num_visible_units, dtype=tf.int8)
        visible_zeros = tf.zeros(self.num_visible_units, dtype=tf.int8)
        term_time = 0
        for i in range(iterations):
            # Positive phase
            pplus = tf.zeros(self.num_connections)

            for p in range(num_patterns):
                if time.time() > term_time + 1 or p == num_patterns - 1:
                    term_time = time.time()
                    sys.stdout.write(f'epoch {i+1}/{iterations}, pattern {p+1}/{num_patterns} clamped         \r')
                    sys.stdout.flush()

                # Setting visible units values
                self.states[0:self.num_visible_units].assign(self.add_noise(patterns[p], noise_probability, noise_bias))

                # Give random values to hidden units
                self.states[-self.num_hidden_units:].assign(np.random.choice([0,1],self.num_hidden_units))

                self.anneal(self.annealing_schedule, visible_ones)
                pplus += self.sum_coocurrance(visible_ones)

            pplus/= trials

            # Negative phase
            sys.stdout.write(f'epoch {i+1}/{iterations}, free running                        \r')
            sys.stdout.flush()
            self.states = tf_random_choice(np.array([0,1]), size=self.states.shape)
            self.anneal(self.annealing_schedule, visible_zeros)
            pminus = self.sum_coocurrance(visible_zeros) / self.coocurrance_cycle.epochs

            self.update_weights(pplus, pminus)
        sys.stdout.write(f'\n')


    def recall(self, pattern, clamp_mask, output_mask=[]):
        # Check if the given pattern hast as many values as clamp_mask has clamped units
        pattern = np.array(pattern)
        if(pattern.shape[0] != tf.nonzero(clamp_mask)[0].shape[0]):
            print("Error: number of given values for recall pattern does not match number of clamped units in given clamp_mask. Exiting.")
            print(f'Got pattern shape {pattern.shape}, clamped units ({tf.nonzero(clamp_mask)[0].shape[0]})')
            sys.exit(1)

        # Set the given clamped units states
        clamped_idxs = tf.where(clamp_mask == 1)[0]
        for i in range(clamped_idxs.shape[0]):
            self.states[clamped_idxs[i]].assign(pattern[i])

        # Set unclamped units states to random 0,1 values
        unclamped_idxs = tf.append(tf.where(clamp_mask == 0)[0], \
                tf.arange(self.num_visible_units, self.num_units))
        self.states[unclamped_idxs].assign(tf.random.choice([0,1], unclamped_idxs.shape[0]))

        self.anneal(self.annealing_schedule, clamp_mask)

        # Decide which states will be output based on the output_mask
        if len(output_mask) == 0:
            # Return all visible units by default
            return self.states[0:self.num_visible_units]
        else:
            # Only return states that are indicated by the output_mask
            output_mask = tf.array(output_mask)
            return self.states[tf.where(output_mask == 1)]


    def add_noise(self, pattern, noise_probability, noise_bias):
        probabilities = noise_probability * pattern + noise_bias
        uniform = tf.random.uniform(shape=[self.num_visible_units])
        return tf.cast(uniform < probabilities, tf.float64)


    def anneal(self, schedule, clamp_mask):
        for step in schedule:
            for epoch in range(step.epochs):
                self.propagate(step.temperature, clamp_mask)


    def propagate(self, temperature, clamp_mask):
        clamp_mask = tf.concat([clamp_mask, tf.zeros(self.num_hidden_units, dtype=tf.int8)], 0)
        unclamped_idxs = tf.where(clamp_mask == 0)[0]

        if (self.synchron_update == True):
            # TODO Some error in here? getting overflow in np.exp Warning
            # might be depending on parameters. synchronous update might need different
            # runtime parameters to work here. It works with energy being dtype np.float128
            # But results using this update function are not good.

            # tf has no dot product, could use matmul with one transposed, or:
            self.energy[unclamped_idxs].assign(tf.reduce_sum(self.weights[unclamped_idxs,:] * tf.cast(self.states, dtype=tf.float64)))
            p = 1. / (1. + tf.exp(-self.energy[unclamped_idxs] / temperature))
            self.states[unclamped_idxs].assign(tf.random.uniform(shape=unclamped_idxs.shape) <= p)
        else:
            # without replacement, every uint will get updated:
            choices = tf.random.shuffle(unclamped_idxs)

            # with replacement:
            # choices = tf_random_choice(unclamped_idxs, size=unclamped_idxs.shape)

            for i in range(unclamped_idxs.shape[0]):
                # Calculating the energy of a randomly selected unit
                unit = choices[i]
                # tf has no dot product, could use matmul with one transposed, or:
                self.energy[unit].assign(tf.reduce_sum(self.weights[unit,:] * tf.cast(self.states, dtype=tf.float64)))
                p = 1. / (1. + tf.exp(-self.energy[unit] / temperature))
                self.states[unit].assign(tf.cond(tf.random.uniform(shape=[1], dtype=tf.float64) <= p, lambda: 1., lambda: 0.))


    def sum_coocurrance(self, clamp_mask):
        sums = tf.zeros(self.num_connections)
        # upper right trangle, without diagonal
        triu = np.triu_indices(self.num_units, 1)
        mask = np.zeros((self.num_units, self.num_units), dtype=bool)
        mask[triu] = True
        for epoch in range(self.coocurrance_cycle.epochs):
            self.propagate(self.coocurrance_cycle.temperature, clamp_mask)
            #mask = triu_mask
            # keep only mask where state of that row == 1
            mask = mask & tf.tile(tf.expand_dims(self.states > 0, 1), [1, self.num_units])
            # keep only mask where state of that col == 1
            mask = mask & tf.tile(tf.expand_dims(self.states > 0, 0), [self.num_units, 1])
            # keep only mask where conn > -1
            mask = mask & (self.connections > -1)
            # indices in sums to update are content of connections
            sum_idx = tf.reshape(self.connections[mask], [-1, 1])
            # generate a sparse vector with dense shape like sums, but 1 only at sum_idx
            add = tf.SparseTensor(tf.cast(sum_idx, dtype=tf.int64), tf.ones(sum_idx.shape[0]), sums.shape)
            # update sums
            sums += tf.sparse.to_dense(add)
        return sums


    def update_weights(self, pplus, pminus):
        # upper right trangle, without diagonal
        triu = tf.triu_indices(self.num_units, 1)
        conn = self.connections[triu]
        filter = conn > -1
        idx = (triu[0][filter], triu[1][filter])
        self.weights[idx].assign(self.weights[idx] + 2*tf.sign(pplus[conn[filter]] - pminus[conn[filter]]))
        # make symmetric, whole wights again
        # cant use np.tril_indices here, would have wrong order
        anti_triu = (triu[1], triu[0])
        self.weights[anti_triu].assign(self.weights[triu])


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
