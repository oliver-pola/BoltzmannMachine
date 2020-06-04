#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mnist_helper import *
from boltzmann import Boltzmann
from boltzmann import load_boltzmann

bm = load_boltzmann(".")

clamp_mask = np.zeros(bm.num_visible_units)

title = f'loaded from files'

path = f'./loaded.png'

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
