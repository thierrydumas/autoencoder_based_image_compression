"""A script to analyze a trained entropy autoencoder by activating one latent variable and deactivating the others.

No dataset is needed for the analysis.

"""

import argparse
import numpy
import os
import tensorflow as tf

import eae.graph.constants as csts
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder
from eae.graph.IsolatedDecoder import IsolatedDecoder

def activating_eae(isolated_decoder, h_in, w_in, bin_widths, idx_map, activation,
                   path_to_restore, path_to_map_mean, path_to_crop):
    """Activates one latent variable and deactivates the others.
    
    One latent variable is activated and the others
    are deactivated. Then, the latent variable feature
    maps are quantized. Finally, the quantized latent
    variable feature maps are passed through the decoder
    of the entropy autoencoder.
    
    Parameters
    ----------
    isolated_decoder : IsolatedDecoder
        Decoder of the entropy autoencoder.
    h_in : int
        Height of the images returned by the
        isolated decoder.
    w_in : int
        Width of the images returned by the
        isolated decoder.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths at the end of the
        training.
    idx_map : int
        Index of the latent variable feature map
        containing the activated latent variable.
        Note that the activated latent variable is
        at the position [1, 1] in this map.
    activation : float
        Activation value.
    path_to_restore : str
        Path to the model to be restored. The
        path must end with ".ckpt".
    path_to_map_mean : str
        Path to the file in which the latent
        variable feature map means are saved.
        The path must end with ".npy".
    path_to_crop : str
        Path to the saved crop of the decoder
        output. The path must end with ".png".
    
    """
    map_mean = numpy.load(path_to_map_mean)
    with tf.Session() as sess:
        isolated_decoder.initialization(sess, path_to_restore)
        y_float32 = numpy.tile(numpy.reshape(map_mean, (1, 1, 1, csts.NB_MAPS_3)),
                               (1, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, 1))
        y_float32[0, 1, 1, idx_map] = activation
        quantized_y_float32 = tls.quantize_per_map(y_float32, bin_widths)
        reconstruction_float32 = sess.run(
            isolated_decoder.node_reconstruction,
            feed_dict={isolated_decoder.node_quantized_y:quantized_y_float32}
        )
        reconstruction_uint8 = numpy.squeeze(tls.cast_bt601(reconstruction_float32), axis=(0, 3))
        tls.save_image(path_to_crop,
                       reconstruction_uint8[0:64, 0:64])

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
    parser.add_argument('--idx_map',
                        help='index of the latent variable feature map containing the activated latent variable',
                        type=parsing.parsing.int_positive,
                        default=49,
                        metavar='')
    parser.add_argument('--activation',
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
                                          suffix_idx_training)
    if not os.path.exists(path_to_directory_crop):
        os.makedirs(path_to_directory_crop)
    path_to_crop = os.path.join(path_to_directory_crop,
                                'activating_map_{0}_{1}.png'.format(args.idx_map + 1, tls.float_to_str(args.activation)))
    
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
    activating_eae(isolated_decoder,
                   h_in,
                   w_in,
                   bin_widths,
                   args.idx_map,
                   args.activation,
                   path_to_restore,
                   path_to_map_mean,
                   path_to_crop)


