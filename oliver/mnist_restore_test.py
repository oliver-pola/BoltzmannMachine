#!/usr/bin/env python

import os
import wget
import gzip
import struct
import shutil
import numpy as np
import matplotlib.pyplot as plt

import mnist
from boltzmann import Boltzmann, load_boltzmann


def test_restore345(images, labels, iterations, num_images, annealing, coocurrence, synchron_update, noise_probability, save_path):
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
    # print(f'learning images shape = {img123.shape}')
    count = img123.shape[0]
    length = img123.shape[1] * img123.shape[2]
    img123 = mnist.im2vec(img123)
    # print(f'learning flattened shape = {img123.shape}')

    # some data statistics: count and mean image (distribution)
    title = f'data distribution for test restore, {count} images'
    plotfilename = save_path + title.replace(',', '').replace(' ', '_') + '.png'
    if not os.path.exists(plotfilename):
        print(f'mean image pixel sum = {np.mean(np.sum(img123, axis=1))}')
        print(f'mean pixel value = {np.mean(img123)}')
        plt.figure(title.replace(',', ''), figsize=(12, 4))
        for i, pick_label in enumerate(pick_labels):
            mean = np.mean(img123[lbl123 == pick_label], axis=0).reshape(images.shape[1], images.shape[2])
            plt.subplot(1, 4, i+1)
            plt.imshow(mean, cmap='gray')
            # if i > 0:
            plt.yticks([])
            plt.xticks([0, 27], [' ', ' '])
            plt.xlabel(f'mean of label {pick_label}', fontsize=16)
        plt.suptitle(title, y=0.88)
        ax = plt.subplot(144)
        ax.hist(lbl123, bins=np.append(pick_labels, np.max(pick_labels) + 1), align='left', rwidth=0.9)
        ax.set_aspect(1.0/ax.get_data_ratio())
        plt.xlabel('label histogram', fontsize=16)
        plt.tight_layout()
        plt.savefig(plotfilename)
        plt.close()

    # consider whole image as input, no output
    out_len = 0
    hidden_layers = length # equal amount as visible layers
    learn_epochs = np.sum(np.array(annealing, dtype=np.int), axis=0)[1]
    cooccur_epochs = coocurrence[1]
    sync_text = 'sync' if synchron_update else 'async'
    title = f'test restore, {iterations} iterations, {count} images, {noise_probability} noise, {hidden_layers} hidden layer, {learn_epochs} learn epochs, {cooccur_epochs} cooccur epochs, T={annealing[-1][0]:.0f}, {sync_text}'
    boltzmannpath = save_path + title.replace(',', '').replace(' ', '_') # folder
    plotfilename = boltzmannpath + '.png' # next to folder
    restorefilename = boltzmannpath + '/restores_mean' # additional file in BM folder

    if os.path.exists(plotfilename) and os.path.exists(restorefilename):
        print('results ready')
    elif os.path.exists(boltzmannpath):
        bm = load_boltzmann(boltzmannpath)
        print('loaded pretrained')
    else:
        # test if pretrained could be continued
        found = False
        for i in range(iterations - 1, 0, -1):
            i_title = f'test restore, {i} iterations, {count} images, {noise_probability} noise, {hidden_layers} hidden layer, {learn_epochs} learn epochs, {cooccur_epochs} cooccur epochs, T={annealing[-1][0]:.0f}, {sync_text}'
            i_path = save_path + i_title.replace(',', '').replace(' ', '_') # folder
            if os.path.exists(i_path):
                found = True
                bm = load_boltzmann(i_path)
                print(f'loaded pretrained with {i} iterations, continue learning')
                bm.learn(img123, iterations - i, noise_probability=noise_probability, reset=False)
                print('continued learning finished')
                bm.save(boltzmannpath)
                break
        # train from scratch
        if not found:
            bm = Boltzmann(length, hidden_layers, out_len,
                annealing, coocurrence, synchron_update=synchron_update)
            # learning
            bm.learn(img123, iterations, noise_probability=noise_probability)
            print('learning finished')
            bm.save(boltzmannpath)

    # pick a sample 3
    sample = images[labels == 3][0]
    # destroy lower half just for visualization
    original = mnist.im2vec(sample[images.shape[1] // 2:, :])
    # reduce to the remaining half and flatten
    destroyed = mnist.im2vec(sample[:images.shape[1] // 2, :])
    # recall from Boltzmann
    clamp_mask = np.append(np.ones(length // 2), np.zeros(length // 2))
    output_mask = np.append(np.zeros(length // 2), np.ones(length // 2))

    if not os.path.exists(plotfilename):
        restore = bm.recall(destroyed, clamp_mask, output_mask)
        plt.figure(title.replace(',', ''), figsize=(12, 4))
        plt.subplot(141)
        plt.imshow(sample, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('sample', fontsize=16)
        sample[images.shape[1] // 2:, :] = 0
        plt.subplot(142)
        plt.imshow(sample, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('destroyed', fontsize=16)
        plt.suptitle(title, y=0.91)
        # fill the lower half of sample
        sample[images.shape[1] // 2:, :] = restore.reshape(-1, images.shape[2])
        plt.subplot(143)
        plt.imshow(sample, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('restored', fontsize=16)

    # try multiple restores and take average
    if os.path.exists(restorefilename):
        restore = np.loadtxt(restorefilename, dtype=np.float)
        print('restore loaded')
    else:
        print('restore by recall...')
        restores = [bm.recall(destroyed, clamp_mask, output_mask) for i in range(20)]
        restore = np.mean(np.array(restores, dtype=np.float), axis=0)
        np.savetxt(restorefilename, restore)

    if not os.path.exists(plotfilename):
        sample[images.shape[1] // 2:, :] = restore.reshape(-1, images.shape[2])
        plt.subplot(144)
        plt.imshow(sample, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('mean of 20 restores', fontsize=16)
        plt.tight_layout()
        plt.savefig(plotfilename)
        plt.close()

    # some measure about the quality of the restore
    good_pixels = np.sum(restore[original == 1])
    bad_pixels = np.sum(restore[original == 0])
    print(f'good_pixels = {good_pixels}, bad_pixels = {bad_pixels}')
    quality = good_pixels - bad_pixels
    return hidden_layers, quality


def plt_quality(title, xlabel, x, qualities, save_path, xticks=None, legend=None):
    plt.figure(title.replace(',', ''), figsize=(12, 4))
    for quality in qualities:
        plt.plot(x, quality, 'o-')
    plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    if not xticks is None:
        plt.xticks(xticks)
    plt.ylabel('quality q', fontsize=16)
    if not legend is None:
        plt.legend(legend, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path + title.replace(',', '').replace(' ', '_') + '.png')
    # plt.close()


def plt_quality_sep_y(title, xlabel, x, qualities, save_path, xticks=None, legend=None):
    fig = plt.figure(title.replace(',', ''), figsize=(12, 4))
    ax1 = fig.add_subplot(111)
    axi = None
    for i, quality in enumerate(qualities):
        if axi is None:
            axi = ax1
            plts = axi.plot(x, quality, 'o-', color=f'C{i}')
        else:
            axi = ax1.twinx()
            plts = plts + axi.plot(x, quality, 'o-', color=f'C{i}')
        axi.set_ylabel('quality q', fontsize=16, color=f'C{i}')
        axi.tick_params(axis='y', labelcolor=f'C{i}')
    plt.title(title)
    ax1.set_xlabel(xlabel, fontsize=16)
    if not xticks is None:
        ax1.set_xticks(xticks)
    if not legend is None:
        ax1.legend(plts, legend, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path + title.replace(',', '').replace(' ', '_') + '.png')
    # plt.close()


def mnist_restore_test():
    images, labels = mnist.get_data()
    images = mnist.im2binary(images)
    # plt.imshow(images[0,:,:], cmap='gray')

    plt.rcParams.update({'font.size': 11})
    save_path = '../data/mnist_restore_test/'
    os.makedirs(save_path, exist_ok=True)

    # base parameters
    iterations = 100
    num_images = 10
    annealing = [(8., 100)]
    coocurrence = (annealing[-1][0], 10)
    synchron_update = True
    noise_probability = 0.8
    # test_restore345(images, labels, iterations, num_images, annealing, coocurrence, synchron_update, noise_probability, save_path)

    # variate number of iterations
    quality_sync = []
    quality_async = []
    iterationslist = np.arange(10, 151, 10)
    for iterations in iterationslist:
        print(f'iterations = {iterations}, sync')
        hidden_layers, q = test_restore345(images, labels, iterations, num_images, annealing, coocurrence, True, noise_probability, save_path)
        quality_sync.append(q)
        print(f'iterations = {iterations}, async')
        hidden_layers, q = test_restore345(images, labels, iterations, num_images, annealing, coocurrence, False, noise_probability, save_path)
        quality_async.append(q)
    learn_epochs = np.sum(np.array(annealing, dtype=np.int), axis=0)[1]
    cooccur_epochs = coocurrence[1]
    iterations_as_epochs = iterationslist * (learn_epochs + cooccur_epochs)
    # sync_text = 'sync' if synchron_update else 'async'
    title = f'restore quality, {num_images} images, {noise_probability} noise, {hidden_layers} hidden layer, {learn_epochs} learn epochs, {cooccur_epochs} cooccur epochs, T={annealing[-1][0]:.0f}'
    # plt_quality(title, 'iterations', iterationslist, quality, save_path, xticks=iterationslist)
    plt_quality(title + ', sync', 'epochs', iterations_as_epochs, [quality_sync], save_path)
    plt_quality(title + ', async', 'epochs', iterations_as_epochs, [quality_async], save_path)
    plt_quality(title, 'epochs', iterations_as_epochs, [quality_sync, quality_async], save_path, legend=['synchron', 'asynchron'])

    # variate temperature
    iterations = 100
    quality_sync = []
    quality_async = []
    temps = np.arange(1, 21, 1)
    scan = True # set to False if all data already exists and to just regenerate plot
    for T in temps:
        annealing = [(float(T), annealing[-1][1])]
        coocurrence = (annealing[-1][0], coocurrence[1])
        iterationslist = np.arange(10, iterations + 1, 10)
        if scan:
            # just generate the intermediate plots and keep the final iterations
            for i in iterationslist:
                print(f'T = {T}, iterations = {i}')
                hidden_layers, q = test_restore345(images, labels, i, num_images, annealing, coocurrence, True, noise_probability, save_path)
        else:
            print(f'T = {T}, iterations = {iterations}')
            hidden_layers, q = test_restore345(images, labels, iterations, num_images, annealing, coocurrence, True, noise_probability, save_path)
        quality_sync.append(q)
        if scan:
            for i in iterationslist:
                print(f'T = {T}, iterations = {i}')
                hidden_layers, q = test_restore345(images, labels, i, num_images, annealing, coocurrence, False, noise_probability, save_path)
        else:
            print(f'T = {T}, iterations = {iterations}')
            hidden_layers, q = test_restore345(images, labels, iterations, num_images, annealing, coocurrence, False, noise_probability, save_path)
        quality_async.append(q)
    learn_epochs = np.sum(np.array(annealing, dtype=np.int), axis=0)[1]
    cooccur_epochs = coocurrence[1]
    # sync_text = 'sync' if synchron_update else 'async'
    title = f'restore quality, {iterations} iterations, {num_images} images, {noise_probability} noise, {hidden_layers} hidden layer, {learn_epochs} learn epochs, {cooccur_epochs} cooccur epochs'
    plt_quality(title + ', sync', 'temperature T', temps, [quality_sync], save_path, xticks=temps)
    plt_quality(title + ', async', 'temperature T', temps, [quality_async], save_path, xticks=temps)
    plt_quality_sep_y(title, 'temperature T', temps, [quality_sync, quality_async], save_path, xticks=temps, legend=['synchron', 'asynchron'])

    # plt.show()


if __name__ == '__main__':
    mnist_restore_test()
