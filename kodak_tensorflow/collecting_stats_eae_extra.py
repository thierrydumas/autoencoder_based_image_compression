"""A script to collect statistics on the latent variable feature maps in a trained entropy autoencoder.

The extra set is used to compute these statistics.
At test time, the statistics are independent of the
images in the Kodak test set. The statistics are not
transmitted from the encoder to the decoder: they
incur no coding cost.

"""

import argparse
import numpy
import os
import tensorflow as tf

import lossless.stats
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects statistics on the latent variable feature maps in a trained entropy autoencoder.')
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
    parser.add_argument('--truncated_unary_length',
                        help='length of the truncated unary prefix',
                        type=parsing.parsing.int_strictly_positive,
                        default=10,
                        metavar='')
    args = parser.parse_args()
    
    batch_size = 20
    multipliers = numpy.array([1., 1.25, 1.5, 2., 3., 4., 6., 8., 10.], dtype=numpy.float32)
    path_to_extra = 'lossless/results/extra_data.npy'
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    path_to_nb_itvs_per_side_load = 'eae/results/{0}/nb_itvs_per_side_{1}.pkl'.format(suffix, args.idx_training)
    path_to_restore = 'eae/results/{0}/model_{1}.ckpt'.format(suffix, args.idx_training)
    path_to_stats = 'lossless/results/{0}/training_index_{1}/'.format(suffix, args.idx_training)
    
    # The directory containing the statistics on the
    # latent variable feature maps is created if it
    # does not exist.
    if not os.path.exists(path_to_stats):
        os.makedirs(path_to_stats)
    path_to_map_mean = os.path.join(path_to_stats,
                                    'map_mean.npy')
    path_to_idx_map_exception = os.path.join(path_to_stats,
                                             'idx_map_exception.pkl')
    paths_to_binary_probabilities = [
        os.path.join(path_to_stats, 'binary_probabilities_{}.npy'.format(tls.float_to_str(multipliers[i].item()))) for i in range(multipliers.size)
    ]
    
    # `extra_uint8.dtype` is equal to `numpy.uint8`.
    # The 4th dimension of `extra_uint8` is equal to 1.
    extra_uint8 = numpy.load(path_to_extra)
    (_, h_in, w_in, _) = extra_uint8.shape
    entropy_ae = EntropyAutoencoder(batch_size,
                                    h_in,
                                    w_in,
                                    args.bin_width_init,
                                    args.gamma_scaling,
                                    path_to_nb_itvs_per_side_load,
                                    args.learn_bin_widths)
    with tf.Session() as sess:
        entropy_ae.initialization(sess, path_to_restore)
        lossless.stats.save_statistics(extra_uint8,
                                       sess,
                                       entropy_ae,
                                       batch_size,
                                       multipliers,
                                       args.truncated_unary_length,
                                       path_to_map_mean,
                                       path_to_idx_map_exception,
                                       paths_to_binary_probabilities)


