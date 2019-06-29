"""A script to create the BSDS test set."""

import argparse
import numpy
import os

import datasets.bsds.bsds
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the BSDS test set.')
    parser.add_argument('path_to_folder_rgbs',
                        help='path to the folder storing the original BSDS dataset')
    parser.add_argument('--path_to_folder_tar',
                        help='path to the folder storing the downloaded archive containing the original BSDS dataset',
                        default='',
                        metavar='')
    args = parser.parse_args()
    
    path_to_bsds = 'datasets/bsds/results/bsds.npy'
    path_to_list_rotation = 'datasets/bsds/results/list_rotation.pkl'
    
    # If `args.path_to_folder_tar` is equal to '', this
    # means that there is no need to download and extract
    # the archive "BSDS300-images.tgz".
    if args.path_to_folder_tar:
        path_to_tar = os.path.join(args.path_to_folder_tar,
                                   'BSDS300-images.tgz')
    else:
        path_to_tar = ''
    datasets.bsds.bsds.create_bsds('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz',
                                   args.path_to_folder_rgbs,
                                   path_to_bsds,
                                   path_to_list_rotation,
                                   path_to_tar=path_to_tar)
    
    # Two luminance images from the BSDS test set are checked visually.
    reference_uint8 = numpy.load(path_to_bsds)
    tls.save_image('datasets/bsds/visualization/luminance_7.png',
                   reference_uint8[7, :, :])
    tls.save_image('datasets/bsds/visualization/luminance_39.png',
                   reference_uint8[39, :, :])


