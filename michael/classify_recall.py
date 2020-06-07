#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mnist_helper import *
from boltzmann import Boltzmann


def classify_image(image, input_images):
    best_match = -1
    best_val = image.shape[0]*image.shape[1]

    for i in range(0,input_images.shape[0]):
        count = np.count_nonzero(input_images[i,:,:] != image)
        if count < best_val:
            best_match = i
            best_val = count

    print(f'Best matching images has hamming distance of {best_val}')
    fig, axarr = plt.subplots(1,2)
    fig.suptitle(f'Hamming distance {best_val}')

    axarr[0].set_title("recall image")
    axarr[0].imshow(image, cmap="gray")
    axarr[1].set_title("original image")
    axarr[1].imshow(input_images[best_match,:,:], cmap="gray")
    plt.show()



images, labels = get_data()
mask1 = (labels == 1)
mask2 = (labels == 2)
mask3 = (labels == 3)
ims1 = images[mask1][:3,:,:]
ims2 = images[mask2][:3,:,:]
ims3 = images[mask3][:3,:,:]
input_images = np.concatenate((ims1,ims2,ims3))
binary_ims = im2binary(input_images)
binary_vecs = im2vec(binary_ims[:,:,:])

annealing_sched = [(10.,100)]
coocurance_cyc = (10.,5)
visible_units = 28*28
hidden_units = 60
iterations = 250
sync_update=False

bm = Boltzmann(visible_units, hidden_units, 0, annealing_sched, coocurance_cyc, sync_update)
bm.learn(binary_vecs, iterations)

clamp_mask = np.zeros(visible_units)

for i in range(5):
    recall_vec = bm.recall([], clamp_mask)
    recall_im = vec2im(recall_vec)
    classify_image(recall_im, binary_ims)





