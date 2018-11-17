"""A script to create the SVHN training set, the SVHN validation set and the SVHN test set."""

import argparse
import numpy

import parsing.parsing
import svhn.svhn
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the SVHN training set, the SVHN validation set and the SVHN test set.')
    parser.add_argument('--nb_training',
                        help='number of SVHN digits in the training set',
                        type=parsing.parsing.int_strictly_positive,
                        default=200000,
                        metavar='')
    parser.add_argument('--nb_validation',
                        help='number of SVHN digits in the validation set',
                        type=parsing.parsing.int_strictly_positive,
                        default=1000,
                        metavar='')
    parser.add_argument('--nb_test',
                        help='number of SVHN digits in the test set',
                        type=parsing.parsing.int_strictly_positive,
                        default=1000,
                        metavar='')
    args = parser.parse_args()
    
    source_url = 'http://ufldl.stanford.edu/housenumbers/'
    path_to_store_mats = 'svhn/data'
    paths_to_outputs = (
        'svhn/results/training_data.npy',
        'svhn/results/validation_data.npy',
        'svhn/results/test_data.npy',
        'svhn/results/mean_training.npy',
        'svhn/results/std_training.npy'
    )
    nb_display = 200
    
    svhn.svhn.create_svhn(source_url,
                          path_to_store_mats,
                          args.nb_training,
                          args.nb_validation,
                          args.nb_test,
                          paths_to_outputs)
    
    # `training_uint8.dtype` is equal to `numpy.uint8`.
    training_uint8 = numpy.load(paths_to_outputs[0])
    mean_training = numpy.load(paths_to_outputs[3])
    std_training = numpy.load(paths_to_outputs[4])
    sample_uint8 = training_uint8[0:nb_display, :]
    
    # The function `svhn.svhn.preprocess_svhn` checks
    # that `sample_uint8.dtype` is equal to `numpy.uint8`
    # and `sample_uint8.ndim` is equal to 2.
    sample_float64 = svhn.svhn.preprocess_svhn(sample_uint8,
                                               mean_training,
                                               std_training)
    tls.visualize_rows(sample_uint8,
                       32,
                       32,
                       10,
                       'svhn/visualization/sample_training.png')
    mu = numpy.mean(sample_float64, axis=0)
    sigma = numpy.sqrt(numpy.mean((sample_float64 - numpy.tile(mu, (nb_display, 1)))**2, axis=0))
    tls.histogram(mu,
                  'Mean of each pixel over {} training images after preprocessing'.format(nb_display),
                  'svhn/visualization/mean_after_preprocessing.png')
    tls.histogram(sigma,
                  'Std of each pixel over {} training images after preprocessing'.format(nb_display),
                  'svhn/visualization/std_after_preprocessing.png')


