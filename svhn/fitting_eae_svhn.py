"""A script to fit a Laplace density to the normed histogram of the latent variables in a trained entropy autoencoder.

250 digits from the SVHN test set are used
for the fitting.

"""

import argparse
import numpy
import os
import pickle

import eae.analysis
import parsing.parsing
import svhn.svhn
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fits a Laplace density to the normed histogram of the latent variables in a trained entropy autoencoder.')
    parser.add_argument('bin_width_init',
                        help='value of the quantization bin width at the beginning of the training',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('gamma',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('--learn_bin_width',
                        help='if given, at training time, the quantization bin width was learned',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    
    path_to_test = 'svhn/results/test_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    if args.learn_bin_width:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                              tls.float_to_str(args.gamma))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                  tls.float_to_str(args.gamma))
    path_to_checking_f = os.path.join('eae/visualization/test/checking_fitting/',
                                      suffix)
    if not os.path.isdir(path_to_checking_f):
        os.makedirs(path_to_checking_f)
    path_to_model = 'eae/results/eae_svhn_{}.pkl'.format(suffix)
    
    # `reference_uint8.dtype` is equal to `numpy.uint8`.
    reference_uint8 = numpy.load(path_to_test)[0:250, :]
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    
    # The function `svhn.svhn.preprocess_svhn` checks
    # that `reference_uint8.dtype` is equal to `numpy.uint8`
    # and `reference_uint8.ndim` is equal to 2.
    reference_float64 = svhn.svhn.preprocess_svhn(reference_uint8,
                                                  mean_training,
                                                  std_training)
    with open(path_to_model, 'rb') as file:
        entropy_ae = pickle.load(file)
    eae.analysis.fit_latent_variables(reference_float64,
                                      entropy_ae,
                                      'Latent variables',
                                      os.path.join(path_to_checking_f, 'fitting_laplace.png'))


