#!/usr/bin/env python

import os
import wget
import gzip
import struct
import shutil
import numpy as np
import matplotlib.pyplot as plt


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
    images_gz = 'data/mnist/train-images-idx3-ubyte.gz'
    images_file = 'data/mnist/train-images-idx3-ubyte'
    labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    labels_gz = 'data/mnist/train-labels-idx1-ubyte.gz'
    labels_file = 'data/mnist/train-labels-idx1-ubyte'
    os.makedirs('data/mnist', exist_ok=True)
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


def im2vec(images):
    if images.ndim == 2:
        length = images.shape[0] * images.shape[1]
        return images.reshape(length)
    elif images.ndim == 3:
        count = images.shape[0]
        length = images.shape[1] * images.shape[2]
        return images.reshape(count, length)


def vec2im(vectors):
    if vectors.ndim == 1:
        length = np.sqrt(vectors.shape[0]).astype(np.int)
        return vectors.reshape((length,length))
    elif vectors.ndim == 2:
        count = vectors.shape[0]
        length = np.sqrt(vectors.shape[1]).astype(np.int)
        return vectors.reshape((count,length,length))


def im2binary(images):
    return (images > 0.5)*1
