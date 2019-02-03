"""A script to test a function for creating the ImageNet training and validation sets."""

import argparse
import numpy
import os

import datasets.imagenet.imagenet
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests a function for creating the ImageNet training and validation sets.')
    parser.parse_args()
    
    # A 1st image is saved at
    # "datasets/imagenet/pseudo_visualization/create_imagenet/training_0.png".
    # A 2nd image is saved at
    # "datasets/imagenet/pseudo_visualization/create_imagenet/training_1.png".
    # A 3rd image is saved at
    # "datasets/imagenet/pseudo_visualization/create_imagenet/validation_0.png".
    # A 4th image is saved at
    # "datasets/imagenet/pseudo_visualization/create_imagenet/validation_1.png".
    # The test is successful if each saved image is
    # the luminance crop of a different RGB image in
    # the folder "datasets/imagenet/pseudo_data/". Besides,
    # if the saved image is named "training_...", it must
    # be a random crop. If the saved image is named
    # "validation_...", it must be a central crop.
    width_crop = 192
    nb_training = 2
    nb_validation = 2
    path_to_training = 'datasets/imagenet/pseudo_data/pseudo_training_data.npy'
    path_to_validation = 'datasets/imagenet/pseudo_data/pseudo_validation_data.npy'
    
    # The images in the folder "datasets/imagenet/pseudo_data/"
    # are large. Therefore, none of them is dumped during
    # the preprocessing.
    datasets.imagenet.imagenet.create_imagenet('datasets/imagenet/pseudo_data/',
                                               width_crop,
                                               nb_training,
                                               nb_validation,
                                               path_to_training,
                                               path_to_validation)
    pseudo_training_data = numpy.load(path_to_training)
    path_to_pseudo_vis = 'datasets/imagenet/pseudo_visualization/create_imagenet/'
    for i in range(nb_training):
        tls.save_image(os.path.join(path_to_pseudo_vis, 'training_{}.png'.format(i)),
                       pseudo_training_data[i, :, :, 0])
    pseudo_validation_data = numpy.load(path_to_validation)
    for i in range(nb_validation):
        tls.save_image(os.path.join(path_to_pseudo_vis, 'validation_{}.png'.format(i)),
                       pseudo_validation_data[i, :, :, 0])


