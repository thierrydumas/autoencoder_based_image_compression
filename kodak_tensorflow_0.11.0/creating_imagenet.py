"""A script to create the ImageNet training and validation sets."""

import argparse
import numpy

import imagenet.imagenet
import parsing.parsing
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the ImageNet training and validation sets.')
    parser.add_argument('path_to_root',
                        help='path to the folder storing ImageNet RGB images')
    parser.add_argument('--width_crop',
                        help='width of the crop',
                        type=parsing.parsing.int_strictly_positive,
                        default=256,
                        metavar='')
    parser.add_argument('--nb_training',
                        help='number of luminance crops in the ImageNet training set',
                        type=parsing.parsing.int_strictly_positive,
                        default=24000,
                        metavar='')
    parser.add_argument('--nb_validation',
                        help='number of luminance crops in the ImageNet validation set',
                        type=parsing.parsing.int_strictly_positive,
                        default=10,
                        metavar='')
    parser.add_argument('--path_to_tar',
                        help='path to the file "ILSVRC2012_img_val.tar", downloaded from <http://image-net.org/download>',
                        default='',
                        metavar='')
    args = parser.parse_args()
    path_to_training = 'imagenet/results/training_data.npy'
    path_to_validation = 'imagenet/results/validation_data.npy'
    
    imagenet.imagenet.create_imagenet(args.path_to_root,
                                      args.width_crop,
                                      args.nb_training,
                                      args.nb_validation,
                                      path_to_training,
                                      path_to_validation,
                                      path_to_tar=args.path_to_tar)
    training_uint8 = numpy.load(path_to_training)
    tls.visualize_luminances(training_uint8[0:24, :, :, :],
                             4,
                             'imagenet/visualization/sample_training.png')


