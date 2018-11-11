"""A script to cause an entropy autoencoder to overfit a small portion of the SVHN training set."""

import argparse
import numpy
import os
import time

import eae.eae_utils as eaeuls
import parsing.parsing
import tools.tools as tls
import svhn.svhn
from eae.EntropyAutoencoder import EntropyAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causes an entropy autoencoder to overfit a small portion of the SVHN training set.')
    parser.add_argument('bin_width_init',
                        help='value of the quantization bin width at the beginning of the training',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('gamma',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('--learn_bin_width',
                        help='if given, the quantization bin width is learned',
                        action='store_true',
                        default=False)
    parser.add_argument('--nb_hidden',
                        help='number of encoder hidden units',
                        type=parsing.parsing.int_strictly_positive,
                        default=300,
                        metavar='')
    parser.add_argument('--nb_y',
                        help='number of latent variables',
                        type=parsing.parsing.int_strictly_positive,
                        default=200,
                        metavar='')
    parser.add_argument('--nb_epochs_fitting',
                        help='number of fitting epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=50,
                        metavar='')
    parser.add_argument('--nb_epochs_training',
                        help='number of training epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=5000,
                        metavar='')
    parser.add_argument('--nb_training',
                        help='number of examples in the small portion of the SVHN training set',
                        type=parsing.parsing.int_strictly_positive,
                        default=10,
                        metavar='')
    args = parser.parse_args()
    path_to_training_data = 'svhn/results/training_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    if args.learn_bin_width:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                              tls.float_to_str(args.gamma))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                  tls.float_to_str(args.gamma))
    path_to_checking_l = 'eae/visualization/overfitting/checking_loss/' + suffix + '/'
    if not os.path.exists(path_to_checking_l):
        os.makedirs(path_to_checking_l)
    path_to_checking_c = 'eae/visualization/overfitting/checking_compression/' + suffix + '/'
    if not os.path.exists(path_to_checking_c):
        os.makedirs(path_to_checking_c)
    assert args.nb_epochs_training % 100 == 0, \
        'The number of training epochs is not divisible by 100.'
    nb_measures = 1 + args.nb_epochs_training//100
    
    # `training_uint8.dtype` is equal to `numpy.uint8`.
    training_uint8 = numpy.load(path_to_training_data)[0:args.nb_training, :]
    nb_visible = training_uint8.shape[1]
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    entropy_ae = EntropyAutoencoder(nb_visible,
                                    args.nb_hidden,
                                    args.nb_y,
                                    args.bin_width_init,
                                    args.gamma,
                                    args.learn_bin_width,
                                    lr_eae=1.e-5)
    
    print('\nThe preliminary fitting of the parameters of the piecewise linear function starts.')
    eaeuls.preliminary_fitting(training_uint8,
                               mean_training,
                               std_training,
                               entropy_ae,
                               args.nb_training,
                               args.nb_epochs_fitting)
    print('The preliminary fitting is completed.')
    
    # The function `svhn.svhn.preprocess_svhn` checks
    # that `training_uint8.dtype` is equal to `numpy.uint8`
    # and `training_uint8.ndim` is equal to 2.
    training_float64 = svhn.svhn.preprocess_svhn(training_uint8,
                                                 mean_training,
                                                 std_training)
    scaled_approx_entropy = numpy.zeros((1, nb_measures))
    rec_error = numpy.zeros((1, nb_measures))
    loss_density_approx = numpy.zeros((1, nb_measures))
    counter = 0
    t_start = time.time()
    for i in range(args.nb_epochs_training):
        if i == 0 or (i + 1) % 100 == 0:
            print('\nEpoch: {}'.format(i + 1))
            (_, _, scaled_approx_entropy[0, counter], rec_error[0, counter], loss_density_approx[0, counter], _) = \
                entropy_ae.evaluation(training_float64)
            print('Training scaled approximate entropy: {}'.format(scaled_approx_entropy[0, counter]))
            print('Training reconstruction error: {}'.format(rec_error[0, counter]))
            print('Training loss of density approximation: {}'.format(loss_density_approx[0, counter]))
            print('Quantization bin width: {}'.format(entropy_ae.bin_width))
            counter += 1
        entropy_ae.training_fct(training_float64)
        entropy_ae.training_eae_bw(training_float64)
    
    evenly_spaced = numpy.linspace(100,
                                   args.nb_epochs_training,
                                   num=nb_measures - 1,
                                   dtype=numpy.int32)
    x_values = numpy.concatenate((numpy.ones(1, dtype=numpy.int32), evenly_spaced))
    tls.plot_graphs(x_values,
                    scaled_approx_entropy,
                    'epoch',
                    'scaled approximate entropy of the quantized latent variables',
                    ['training'],
                    ['r'],
                    'Evolution of the scaled approximate entropy over epochs',
                    path_to_checking_l + 'scaled_approximate_entropy.png')
    tls.plot_graphs(x_values,
                    rec_error,
                    'epoch',
                    'reconstruction error',
                    ['training'],
                    ['r'],
                    'Evolution of the reconstruction error over epochs',
                    path_to_checking_l + 'reconstruction_error.png')
    tls.plot_graphs(x_values,
                    loss_density_approx,
                    'epoch',
                    'loss of density approximation',
                    ['training'],
                    ['r'],
                    'Evolution of the loss of density approximation over epochs',
                    path_to_checking_l + 'loss_density_approximation.png')
    t_stop = time.time()
    nb_minutes = int((t_stop - t_start)/60)
    print('\nTraining time: {} minutes.'.format(nb_minutes))
    
    tls.visualize_rows(training_uint8,
                       32,
                       32,
                       2,
                       path_to_checking_c + 'reference.png')
    
    # If, at training time, the quantization bin
    # width was learned, the test quantization bin
    # width is equal to the quantization bin width
    # at the end of the training, rounded to the
    # 1st digit after the decimal point.
    if args.learn_bin_width:
        bin_width_test = round(entropy_ae.bin_width, 1)
    else:
        bin_width_test = args.bin_width_init
    (rate, psnr) = eaeuls.compute_rate_psnr(training_uint8,
                                            mean_training,
                                            std_training,
                                            entropy_ae,
                                            bin_width_test,
                                            2,
                                            path_to_checking_c + 'reconstruction.png')
    print('\nMean rate over the {0} training RGB digits: {1}'.format(args.nb_training, rate))
    print('Mean PSNR over the {0} training RGB digits: {1}'.format(args.nb_training, psnr))


