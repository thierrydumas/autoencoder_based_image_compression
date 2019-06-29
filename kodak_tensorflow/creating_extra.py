"""A script to create the extra set.

The extra set is used to compute statistics on the
latent variable feature maps in different entropy
autoencoders.

"""

import argparse
import numpy
import os
import random

import datasets.extra.extra
import parsing.parsing
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the extra set.')
    parser.add_argument('path_to_folder_rgbs_ilsvrc2012',
                        help='path to the folder storing ImageNet RGB images')
    parser.add_argument('path_to_folder_rgbs_inria_holidays',
                        help='path to the folder storing the original INRIA Holidays dataset')
    parser.add_argument('--width_crop',
                        help='width of the crop',
                        type=parsing.parsing.int_strictly_positive,
                        default=384,
                        metavar='')
    parser.add_argument('--nb_extra',
                        help='number of luminance crops in the extra set',
                        type=parsing.parsing.int_strictly_positive,
                        default=1000,
                        metavar='')
    parser.add_argument('--path_to_tar_ilsvrc2012',
                        help='path to the ILSVRC2012 validation archive, downloaded from <http://image-net.org/download>',
                        default='',
                        metavar='')
    parser.add_argument('--path_to_folder_tar_inria_holidays',
                        help='path to the folder storing the downloaded archive containing the original INRIA Holidays dataset',
                        default='',
                        metavar='')
    args = parser.parse_args()
    
    # The random seed is set as a function called by `datasets.extra.extra.create_extra`
    # involves a shuffling.
    random.seed(0)
    path_to_extra = 'datasets/extra/results/extra_data.npy'
    
    # If `args.path_to_folder_tar_inria_holidays` is equal to '',
    # this means that there is no need to download and extract
    # the archives "jpg1.tar.gz" and "jpg2.tar.gz".
    if args.path_to_folder_tar_inria_holidays:
        path_to_tar_inria_holidays_0 = os.path.join(args.path_to_folder_tar_inria_holidays,
                                                    'jpg1.tar.gz')
        path_to_tar_inria_holidays_1 = os.path.join(args.path_to_folder_tar_inria_holidays,
                                                    'jpg2.tar.gz')
    else:
        path_to_tar_inria_holidays_0 = ''
        path_to_tar_inria_holidays_1 = ''
    datasets.extra.extra.create_extra('ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz',
                                      'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz',
                                      args.path_to_folder_rgbs_ilsvrc2012,
                                      args.path_to_folder_rgbs_inria_holidays,
                                      args.width_crop,
                                      args.nb_extra,
                                      path_to_extra,
                                      path_to_tar_ilsvrc2012=args.path_to_tar_ilsvrc2012,
                                      path_to_tar_inria_holidays_0=path_to_tar_inria_holidays_0,
                                      path_to_tar_inria_holidays_1=path_to_tar_inria_holidays_1)
    
    # Nine luminance images from the extra set are checked visually.
    extra_uint8 = numpy.load(path_to_extra)
    tls.visualize_luminances(extra_uint8[0:9, :, :, :],
                             3,
                             'datasets/extra/visualization/sample_extra.png')


