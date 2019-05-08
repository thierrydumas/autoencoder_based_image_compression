"""A script to analyze a trained entropy autoencoder by masking all the latent variable feature maps except one.

The Kodak test set is used for the analysis.

"""

import argparse
import numpy
import os
import tensorflow as tf

import eae.analysis
import eae.batching
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder
from eae.graph.IsolatedDecoder import IsolatedDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzes a trained entropy autoencoder by masking all the latent variable feature maps except one.')
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
    parser.add_argument('--idx_unmasked_map',
                        help='index of the unmasked latent variable feature map',
                        type=parsing.parsing.int_positive,
                        default=51,
                        metavar='')
    args = parser.parse_args()
    
    batch_size = 4
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    suffix_idx_training = '{0}/training_index_{1}'.format(suffix, args.idx_training)
    path_to_nb_itvs_per_side_load = 'eae/results/{0}/nb_itvs_per_side_{1}.pkl'.format(suffix, args.idx_training)
    path_to_restore = 'eae/results/{0}/model_{1}.ckpt'.format(suffix, args.idx_training)
    path_to_map_mean = 'lossless/results/{}/map_mean.npy'.format(suffix_idx_training)
    path_to_checking_m = 'eae/visualization/test/checking_masking/{0}/unmasked_map_{1}/'.format(suffix_idx_training,
                                                                                                args.idx_unmasked_map + 1)
    if not os.path.isdir(path_to_checking_m):
        os.makedirs(path_to_checking_m)
    reference_uint8 = numpy.load('datasets/kodak/results/kodak.npy')
    (nb_images, h_in, w_in) = reference_uint8.shape
    luminances_uint8 = numpy.expand_dims(reference_uint8, axis=3)
    map_mean = numpy.load(path_to_map_mean)
    
    # A single entropy autoencoder is created.
    entropy_ae = EntropyAutoencoder(batch_size,
                                    h_in,
                                    w_in,
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
        
        # `bin_widths` are the quantization bin widths
        # at the end of the training.
        bin_widths = entropy_ae.get_bin_widths()
    
    # The graph of the entropy autoencoder is destroyed.
    tf.reset_default_graph()
    
    # A single decoder is created.
    isolated_decoder = IsolatedDecoder(1,
                                       h_in,
                                       w_in,
                                       args.learn_bin_widths)
    with tf.Session() as sess:
        isolated_decoder.initialization(sess, path_to_restore)
        eae.analysis.mask_maps(y_float32,
                               sess,
                               isolated_decoder,
                               bin_widths,
                               args.idx_unmasked_map,
                               map_mean,
                               200,
                               200,
                               [os.path.join(path_to_checking_m, 'masking_map_{}.png'.format(i + 1)) for i in range(nb_images)])


