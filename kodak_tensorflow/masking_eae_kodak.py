"""A script to analyze a trained entropy autoencoder by masking all the latent variable feature maps except one.

The Kodak test set is used for the analysis.

"""

import argparse
import numpy
import os
import scipy.misc
import tensorflow as tf

import eae.eae_utils as eaeuls
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder
from eae.graph.IsolatedDecoder import IsolatedDecoder

def masking_eae_kodak(y_float32, isolated_decoder, bin_widths, idx_map, path_to_restore, path_to_map_mean, paths):
    """Masks all the latent variable feature maps except one.
    
    All the latent variable feature maps except one
    are masked. Then, the latent variable feature maps
    are quantized. Finally, the quantized latent
    variable feature maps are passed through the
    decoder of the entropy autoencoder.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variables. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of
        the ith example.
    isolated_decoder : IsolatedDecoder
        Decoder of the entropy autoencoder.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths at the end of the
        training.
    idx_map : int
        Index of the unmasked latent variable
        feature map.
    path_to_restore : str
        Path to the model to be restored. The
        path must end with ".ckpt".
    path_to_map_mean : str
        Path to the file in which the latent
        variable feature map means are saved.
        The path must end with ".npy".
    paths : list
        The ith string in this list is the
        path to the ith saved crop of the
        decoder output. Each path must end
        with ".png".
    
    Raises
    ------
    AssertionError
        If `len(paths)` is not equal to `y_float32.shape[0]`.
    
    """
    nb_images = y_float32.shape[0]
    assert len(paths) == nb_images, '`len(paths)` is not equal to `y_float32.shape[0]`.'
    map_mean = numpy.load(path_to_map_mean)
    with tf.Session() as sess:
        isolated_decoder.initialization(sess, path_to_restore)
        
        # The same latent variable feature map is
        # iteratively overwritten in the loop below.
        masked_y_float32 = numpy.tile(numpy.reshape(map_mean, (1, 1, 1, y_float32.shape[3])),
                                      (1, y_float32.shape[1], y_float32.shape[2], 1))
        for i in range(nb_images):
            masked_y_float32[0, :, :, idx_map] = y_float32[i, :, :, idx_map]
            quantized_y_float32 = tls.quantize_per_map(masked_y_float32, bin_widths)
            reconstruction_float32 = sess.run(
                isolated_decoder.node_reconstruction,
                feed_dict={isolated_decoder.node_quantized_y:quantized_y_float32}
            )
            reconstruction_uint8 = numpy.squeeze(tls.cast_bt601(reconstruction_float32), axis=(0, 3))
            scipy.misc.imsave(paths[i], reconstruction_uint8[0:200, 0:200])

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
    parser.add_argument('--idx_map',
                        help='index of the unmasked latent variable feature map',
                        type=parsing.parsing.int_positive,
                        default=51,
                        metavar='')
    args = parser.parse_args()
    
    batch_size = 4
    path_to_kodak = 'kodak/results/kodak.npy'
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    suffix_idx_training = '{0}/training_index_{1}'.format(suffix, args.idx_training)
    path_to_nb_itvs_per_side_load = 'eae/results/{0}/nb_itvs_per_side_{1}.pkl'.format(suffix, args.idx_training)
    path_to_restore = 'eae/results/{0}/model_{1}.ckpt'.format(suffix, args.idx_training)
    path_to_map_mean = 'lossless/results/{}/map_mean.npy'.format(suffix_idx_training)
    path_to_checking_m = 'eae/visualization/test/checking_masking/{}/'.format(suffix_idx_training)
    if not os.path.exists(path_to_checking_m):
        os.makedirs(path_to_checking_m)
    reference_uint8 = numpy.load(path_to_kodak)
    (nb_images, h_in, w_in) = reference_uint8.shape
    luminances_uint8 = numpy.expand_dims(reference_uint8, axis=3)
    
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
        y_float32 = eaeuls.encode_mini_batches(luminances_uint8,
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
    masking_eae_kodak(y_float32,
                      isolated_decoder,
                      bin_widths,
                      args.idx_map,
                      path_to_restore,
                      path_to_map_mean,
                      [os.path.join(path_to_checking_m, 'masking_map_{0}_{1}.png'.format(args.idx_map + 1, i + 1)) for i in range(nb_images)])


