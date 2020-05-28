#!/usr/bin/env python

import os
import wget
import gzip
import struct
import shutil
import numpy as np
import matplotlib.pyplot as plt

from boltzmann import Boltzmann


def read_idx1(path):
    # https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
    with open(path,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        # np.fromfile can't read a file opened with gzip.open()
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size))
    return data


def read_idx3(path):
    # https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
    with open(path,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        # np.fromfile can't read a file opened with gzip.open()
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
    return data


def gunzip(path_gz, path):
    with gzip.open(path_gz, 'rb') as f_in, open(path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def get_data():
    images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    images_gz = '../data/mnist/train-images-idx3-ubyte.gz'
    images_file = '../data/mnist/train-images-idx3-ubyte'
    labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    labels_gz = '../data/mnist/train-labels-idx1-ubyte.gz'
    labels_file = '../data/mnist/train-labels-idx1-ubyte'
    os.makedirs('../data/mnist', exist_ok=True)
    if not os.path.exists(images_file):
        if not os.path.exists(images_gz):
            print('Downloading MNIST training images...')
            wget.download(images_url, images_gz)
            print()
        print('Unzip MNIST training images...')
        gunzip(images_gz, images_file)
        os.remove(images_gz)
    if not os.path.exists(labels_file):
        if not os.path.exists(labels_gz):
            print('Downloading MNIST training labels...')
            wget.download(labels_url, labels_gz)
            print()
        print('Unzip MNIST training labels...')
        gunzip(labels_gz, labels_file)
        os.remove(labels_gz)
    # images = read_idx3(images_file) / 255 # range [0,1]
    images = (read_idx3(images_file) > 127) * 1. # binary 0 or 1
    labels = read_idx1(labels_file)
    return images, labels


def test_restore345(images, labels, epochs, num_images):
    # just pick images with label 3, 4, 5
    pick_labels = [3, 4, 5]
    mask = np.zeros(labels.shape, dtype=bool)
    for pick_label in pick_labels:
        mask = mask | (labels == pick_label)
    img123 = images[mask, :, :]
    lbl123 = labels[mask]
    # reduce number of images
    if num_images < img123.shape[0]:
        img123 = img123[:num_images, :, :]
        lbl123 = lbl123[:num_images]
    # reshape 28 x 28 image to vector of length 784
    print(f'learning images shape = {img123.shape}')
    count = img123.shape[0]
    length = img123.shape[1] * img123.shape[2]
    img123 = img123.reshape(count, length)
    print(f'learning flattened shape = {img123.shape}')

    # some data statistics: count and mean image (distribution)
    print(f'mean image pixel sum = {np.mean(np.sum(img123, axis=1))}')
    print(f'mean pixel value = {np.mean(img123)}')
    title = f'data distribution for test restore, {count} images'
    plt.figure(title.replace(',', ''), figsize=(12, 4))
    for i, pick_label in enumerate(pick_labels):
        mean = np.mean(img123[lbl123 == pick_label], axis=0).reshape(images.shape[1], images.shape[2])
        plt.subplot(1, 4, i+1)
        plt.imshow(mean, cmap='gray')
        if i > 0:
            plt.yticks([])
        plt.xlabel(f'mean of label {pick_label}')
    plt.suptitle(title, y=0.88)
    ax = plt.subplot(144)
    ax.hist(lbl123, bins=np.append(pick_labels, np.max(pick_labels) + 1), align='left', rwidth=0.9)
    ax.set_aspect(1.0/ax.get_data_ratio())
    plt.xlabel('label histogram')
    plt.tight_layout()

    # consider whole image as input, no output
    out_len = 0
    hidden_layers = length # equal amount as visible layers
    annealing = [(1., 1000)]
    coocurance = (annealing[-1][0], 10)
    synchron_update = True
    noise_probability = 0.5
    b = Boltzmann(length, hidden_layers, out_len,
        annealing, coocurance, synchron_update=True)
    # learning
    b.learn(img123, epochs, noise_probability=noise_probability)
    print('learning finished')

    # pick a sample 3
    sample = images[labels == 3][0]
    blind_iter = np.sum(np.array(annealing, dtype=np.int), axis=0)[1]
    weights_iter = coocurance[1]
    sync_text = 'synchron' if synchron_update else 'asynchron'
    title = f'test restore, {epochs} epochs, {count} images, {noise_probability} noise, {hidden_layers} hidden layer, {blind_iter} blind iter, {weights_iter} weights update iter, T={annealing[-1][0]:.0f}, {sync_text}'
    plt.figure(title.replace(',', ''), figsize=(12, 4))
    plt.subplot(141)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('sample')
    # destroy lower half just for visualization
    sample[images.shape[1] // 2:, :] = 0
    plt.subplot(142)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('destroyed')
    plt.suptitle(title, y=0.91)
    # reduce to the remaining half and flatten
    destroyed = sample[:images.shape[1] // 2, :].reshape(length // 2)
    # recall from Boltzmann
    print('restore by recall...')
    clamp_mask = np.append(np.ones(length // 2), np.zeros(length // 2))
    output_mask = np.append(np.zeros(length // 2), np.ones(length // 2))
    restore = b.recall(destroyed, clamp_mask, output_mask)
    # fill the lower half of sample
    sample[images.shape[1] // 2:, :] = restore.reshape(-1, images.shape[2])
    plt.subplot(143)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('restored')
    # try multiple restores and take average
    restores = [b.recall(destroyed, clamp_mask, output_mask) for i in range(20)]
    restore = np.mean(np.array(restores, dtype=np.float), axis=0)
    sample[images.shape[1] // 2:, :] = restore.reshape(-1, images.shape[2])
    plt.subplot(144)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('mean of 20 restores')
    plt.tight_layout()


def mnist_test():
    images, labels = get_data()
    # plt.imshow(images[0,:,:], cmap='gray')

    # run tests
    epochs = 3
    num_images = 3
    test_restore345(images, labels, epochs, num_images)

    plt.show()


if __name__ == '__main__':
    mnist_test()
