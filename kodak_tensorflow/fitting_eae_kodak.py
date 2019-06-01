"""A script to fit a Laplace density to the normed histogram of each latent variable feature map in a trained entropy autoencoder.

The Kodak test set is used for the fitting.

"""

import argparse
import numpy
import os
import pickle
import tensorflow as tf

import eae.analysis
import eae.batching
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fits a Laplace density to the normed histogram of each latent variable feature map in a trained entropy autoencoder.')
    parser.add_argument('bin_width_init',
                        help='value of the quantization bin widths at the beginning of the 1st training',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('gamma_scaling',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('idx_training',
                        help='training phase index',
                        type=parsing.parsing.int_strictly_positive)
    parser.add_argument('--learn_bin_widths',
                        help='if given, the quantization bin widths were learned at training time',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    
    batch_size = 4
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    suffix_idx_training = '{0}/training_index_{1}'.format(suffix, args.idx_training)
    path_to_nb_itvs_per_side_load = 'eae/results/{0}/nb_itvs_per_side_{1}.pkl'.format(suffix, args.idx_training)
    path_to_restore = 'eae/results/{0}/model_{1}.ckpt'.format(suffix, args.idx_training)
    path_to_idx_map_exception = 'lossless/results/{}/idx_map_exception.pkl'.format(suffix_idx_training)
    path_to_checking_f = 'eae/visualization/test/checking_fitting/{}/'.format(suffix_idx_training)
    
    # The directory containing the normed histogram of
    # each latent variable feature map is created if it
    # does not exist.
    if not os.path.isdir(path_to_checking_f):
        os.makedirs(path_to_checking_f)
    reference_uint8 = numpy.load('datasets/kodak/results/kodak.npy')
    luminances_uint8 = numpy.expand_dims(reference_uint8, axis=3)
    with open(path_to_idx_map_exception, 'rb') as file:
        idx_map_exception = pickle.load(file)
    
    # A single entropy autoencoder is created.
    entropy_ae = EntropyAutoencoder(batch_size,
                                    luminances_uint8.shape[1],
                                    luminances_uint8.shape[2],
                                    args.bin_width_init,
                                    args.gamma_scaling,
                                    path_to_nb_itvs_per_side_load,
                                    args.learn_bin_widths)
    with tf.Session() as sess:
        entropy_ae.initialization(sess, path_to_restore)
        y_float32 = eae.batching.encode_mini_batches(luminances_uint8,
                                                     sess,
                                                     entropy_ae,
                                                     batch_size)
    eae.analysis.fit_maps(y_float32,
                          os.path.join(path_to_checking_f, 'laplace_locations.png'),
                          os.path.join(path_to_checking_f, 'laplace_scales.png'),
                          [os.path.join(path_to_checking_f, 'fitting_map_{}.png'.format(i + 1)) for i in range(y_float32.shape[3])],
                          idx_map_exception=idx_map_exception)


