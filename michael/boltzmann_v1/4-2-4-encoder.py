#!/usr/bin/env python

import numpy as np
from boltzman import Boltzman 

patterns = [[1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1]]

b = Boltzman(8,2,4,[(20.,2),(15.,2),(12.,2),(10.,4)], (10.,10), synchron_update=False)
b.learn(patterns, 1800)

print(b.weights)

clamp_mask = np.append(np.ones(4), np.zeros(4))
output_mask = np.append(np.zeros(4), np.ones(4))
print(b.recall([1,0,0,0], clamp_mask, output_mask))
