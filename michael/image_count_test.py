#!/usr/bin/env python

from runner import recall_image
from mnist_helper import *

def image_count_test():
    images, labels = get_data()
    mask = (labels == 3)
    annealing_sched = [(20.,2),(15.,2),(12.,2),(10.,4)]
    coocurance_cyc = (10.,4)
    hidden_units = 60
    iterations = 100
    out_dir = "./image_count"

    input_images = images[mask][:5,:,:]
    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, iterations,\
            False, 0.8, 0.05, out_dir)

    input_images = images[mask][:10,:,:]
    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, iterations,\
            False, 0.8, 0.05, out_dir)

    input_images = images[mask][:50,:,:]
    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, iterations,\
            False, 0.8, 0.05, out_dir)

    input_images = images[mask][:100,:,:]
    recall_image(input_images, annealing_sched, coocurance_cyc, hidden_units, iterations,\
            False, 0.8, 0.05, out_dir)


image_count_test()
