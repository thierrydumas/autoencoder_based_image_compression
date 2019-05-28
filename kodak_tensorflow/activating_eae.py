"""A script to analyze a trained entropy autoencoder by activating one latent variable and deactivating the others.

No dataset is needed for the analysis.

"""

import argparse
import numpy
import os
import tensorflow as tf

import eae.analysis
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder
from eae.graph.IsolatedDecoder import IsolatedDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzes a trained entropy autoencoder by activating one latent variable and deactivating the others.')
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
    parser.add_argument('--idx_map_activation',
                        help='index of the latent variable feature map containing the activated latent variable',
                        type=parsing.parsing.int_positive,
                        default=49,
                        metavar='')
    parser.add_argument('--activation_value',
                        help='activation value',
                        type=float,
                        default=8.,
                        metavar='')
    args = parser.parse_args()
    
    # The height of the width of the decoder
    # output are not important. They must be
    # larger than 64.
    h_in = 256
    w_in = 256
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    suffix_idx_training = '{0}/training_index_{1}'.format(suffix, args.idx_training)
    path_to_nb_itvs_per_side_load = 'eae/results/{0}/nb_itvs_per_side_{1}.pkl'.format(suffix, args.idx_training)
    path_to_restore = 'eae/results/{0}/model_{1}.ckpt'.format(suffix, args.idx_training)
    path_to_map_mean = 'lossless/results/{}/map_mean.npy'.format(suffix_idx_training)
    path_to_directory_crop = os.path.join('eae/visualization/test/checking_activating/',
                                          suffix_idx_training,
                                          '{0}_{1}'.format(args.idx_map_activation + 1, tls.float_to_str(args.activation_value)))
    if not os.path.isdir(path_to_directory_crop):
        os.makedirs(path_to_directory_crop)
    map_mean = numpy.load(path_to_map_mean)
    
    # A single entropy autoencoder is created.
    entropy_ae = EntropyAutoencoder(1,
                                    h_in,
                                    w_in,
                                    args.bin_width_init,
                                    args.gamma_scaling,
                                    path_to_nb_itvs_per_side_load,
                                    args.learn_bin_widths)
    with tf.Session() as sess:
        entropy_ae.initialization(sess, path_to_restore)
        bin_widths = entropy_ae.get_bin_widths()
    
    # The graph of the entropy autoencoder is destroyed.
    tf.reset_default_graph()
    
    # A single decoder is created.
    isolated_decoder = IsolatedDecoder(1,
                                       h_in,
                                       w_in,
                                       args.learn_bin_widths)
    
    # Different latent variables are successively activated in a feature map
    # to test the translation covariance of the decoder.
    tuple_pairs_row_col = (
        (1, 1),
        (6, 6)
    )
    with tf.Session() as sess:
        isolated_decoder.initialization(sess, path_to_restore)
        for (row_activation, col_activation) in tuple_pairs_row_col:
            path_to_crop = os.path.join(path_to_directory_crop,
                                        'activating_map_{0}_{1}.png'.format(row_activation, col_activation))
            eae.analysis.activate_latent_variable(sess,
                                                  isolated_decoder,
                                                  h_in,
                                                  w_in,
                                                  bin_widths,
                                                  row_activation,
                                                  col_activation,
                                                  args.idx_map_activation,
                                                  args.activation_value,
                                                  map_mean,
                                                  64,
                                                  64,
                                                  path_to_crop)


