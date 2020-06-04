#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mnist_helper import *
from boltzmann import Boltzmann


def recall_image(images, annealing_sched, coocurance_cyc, hidden_units, iterations, synchron, noise_pr, noise_b, out_dir):
    im_vecs = im2vec(images[:,:,:])
    binary_vecs = im2binary(im_vecs)

    visible_units = im_vecs.shape[1]

    bm = Boltzmann(visible_units, hidden_units, 0, annealing_sched, coocurance_cyc)
    bm.learn(binary_vecs, iterations, noise_probability=noise_pr, noise_bias=noise_b)
    clamp_mask = np.zeros(visible_units)

    title = f'images: {str(images.shape[0])}, hidden units: {str(hidden_units)}, iterations: {str(iterations)}, synchron: {str(synchron)}\nannealing shed: {str(annealing_sched)}\nnoise: {str(noise_pr)} , {str(noise_b)}'

    path = f'{out_dir}/{str(images.shape[0])}_{str(hidden_units)}_{str(iterations)}_{str(synchron)}_{str(annealing_sched)}_{str(noise_pr)}_{str(noise_b)}.png'

    fig, axarr = plt.subplots(1,3)
    fig.suptitle(title)
    

    recall_vec = bm.recall([], clamp_mask)
    result_im = vec2im(recall_vec)
    axarr[0].imshow(result_im, cmap='gray')
    recall_vec = bm.recall([], clamp_mask)
    result_im = vec2im(recall_vec)
    axarr[1].imshow(result_im, cmap='gray')
    recall_vec = bm.recall([], clamp_mask)
    result_im = vec2im(recall_vec)
    axarr[2].imshow(result_im, cmap='gray')

    plt.savefig(path)
    bm.save(out_dir)
