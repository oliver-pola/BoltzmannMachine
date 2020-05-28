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
    images = read_idx3(images_file) / 255 # range [0,1]
    labels = read_idx1(labels_file)
    return images, labels


def test_restore345(images, labels, epochs, num_images):
    # just pick images with label 3, 4, 5
    mask = (labels == 3) | (labels == 4) | (labels == 5)
    img123 = images[mask, :, :]
    # reduce number of images
    if num_images < img123.shape[0]:
        img123 = img123[:num_images, :, :]
    # reshape 28 x 28 image to vector of length 784
    print(f'learning images shape = {img123.shape}')
    count = img123.shape[0]
    length = img123.shape[1] * img123.shape[2]
    img123 = img123.reshape(count, length)
    print(f'learning flattened shape = {img123.shape}')
    # consider first half as input, last half as output
    out_len = length // 2
    in_len = length - out_len
    hidden_layers = 64
    b = Boltzmann(length, hidden_layers, out_len,
        [(20.,2),(15.,2),(12.,2),(10.,4)],
        (10.,10), synchron_update=False)
    # learning
    b.learn(img123, epochs)

    # pick a sample 3
    sample = images[labels == 3][0]
    title = f'test restore, {epochs} epochs, {count} images (3s, 4s, 5s)'
    plt.figure(title, figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('sample')
    # destroy lower half just for visualization
    sample[images.shape[1] // 2:, :] = 0
    plt.subplot(132)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('destroyed')
    plt.title(title)
    # reduce to the remaining half and flatten
    destroyed = sample[:images.shape[1] // 2, :].reshape(in_len)
    # recall from Boltzmann
    clamp_mask = np.append(np.ones(in_len), np.zeros(out_len))
    output_mask = np.append(np.zeros(in_len), np.ones(out_len))
    restore = b.recall(destroyed, clamp_mask, output_mask)
    # fill the lower half of sample
    sample[images.shape[1] // 2:, :] = restore.reshape(-1, images.shape[2])
    plt.subplot(133)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('restored')


def mnist_test():
    images, labels = get_data()
    # plt.imshow(images[0,:,:], cmap='gray')

    # run tests
    test_restore345(images, labels, 10, 100)

    plt.show()


if __name__ == '__main__':
    mnist_test()
