#!/usr/bin/env python

from runner import recall_image
from mnist_helper import *

def iteration_test():
    images, labels = get_data()
    mask = (labels == 3)
    input_images = images[mask][:10,:,:]
    annealing_sched = [(20.,2),(15.,2),(12.,2),(10.,4)]
    coocurance_cyc = (10.,4)
    hidden_units = 60
    out_dir = "./iterations"

    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, 100,\
            False, 0.8, 0.05, out_dir)

    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, 200,\
            False, 0.8, 0.05, out_dir)

    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, 500,\
            False, 0.8, 0.05, out_dir)

    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, 1000,\
            False, 0.8, 0.05, out_dir)



iteration_test()
