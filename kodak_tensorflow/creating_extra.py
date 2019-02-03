"""A script to create the extra set.

The extra set is used to compute statistics on the
latent variable feature maps in different entropy
autoencoders.

"""

import argparse
import numpy

import datasets.extra.extra
import parsing.parsing
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the extra set.')
    
    # It is better that the folder at `path_to_root`
    # contain both low resolution and high resolution
    # images. For instance, we put into this folder 300
    # RGB images from the INRIA Holidays dataset
    # <http://lear.inrialpes.fr/~jegou/data.php> and 5000
    # RGB images from the ILSVRC2012 test set
    # <http://image-net.org/download>.
    parser.add_argument('path_to_root',
                        help='path to the folder containing RGB images')
    parser.add_argument('--width_crop',
                        help='width of the crop',
                        type=parsing.parsing.int_strictly_positive,
                        default=384,
                        metavar='')
    parser.add_argument('--nb_extra',
                        help='number of luminance crops in the extra set',
                        type=parsing.parsing.int_strictly_positive,
                        default=600,
                        metavar='')
    args = parser.parse_args()
    
    path_to_extra = 'datasets/extra/results/extra_data.npy'
    
    datasets.extra.extra.create_extra(args.path_to_root,
                                      args.width_crop,
                                      args.nb_extra,
                                      path_to_extra)
    extra_uint8 = numpy.load(path_to_extra)
    
    # The 4th dimension of `extra_uint8` is equal to 1.
    tls.visualize_luminances(extra_uint8[0:9, :, :, :],
                             3,
                             'datasets/extra/visualization/sample_extra.png')


