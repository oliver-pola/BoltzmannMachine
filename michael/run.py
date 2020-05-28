#!/usr/bin/env python

from runner import recall_image
from mnist_helper import *

images, labels = get_data()
mask = (labels == 3)
annealing_sched = [(20.,2),(15.,2),(12.,2),(10.,4)]
coocurance_cyc = (10.,4)
hidden_units = 60
iterations = 100
input_images = images[mask][:10,:,:]
out_dir = "."

recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, iterations,\
        False, 0.8, 0.05, out_dir)


