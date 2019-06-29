"""A script to create the Kodak test set."""

import argparse
import numpy

import datasets.kodak.kodak
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the Kodak test set.')
    parser.parse_args()
    
    path_to_kodak = 'datasets/kodak/results/kodak.npy'
    path_to_list_rotation = 'datasets/kodak/results/list_rotation.pkl'
    
    datasets.kodak.kodak.create_kodak('http://r0k.us/graphics/kodak/kodak/',
                                      'datasets/kodak/data',
                                      path_to_kodak,
                                      path_to_list_rotation)
    
    # Two luminance images from the Kodak test set are checked visually.
    reference_uint8 = numpy.load(path_to_kodak)
    tls.save_image('datasets/kodak/visualization/luminance_3.png',
                   reference_uint8[2, :, :])
    tls.save_image('datasets/kodak/visualization/luminance_11.png',
                   reference_uint8[10, :, :])


