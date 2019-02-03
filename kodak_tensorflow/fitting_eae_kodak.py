"""A script to fit a Laplace density to the normed histogram of each latent variable feature map in a trained entropy autoencoder.

The Kodak test set is used for the fitting.

"""

import argparse
import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import os
import pickle
import scipy.stats
import tensorflow as tf

import eae.eae_utils as eaeuls
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder

def fitting_eae_kodak(y_float32, path_to_idx_map_exception, path_to_checking_f):
    """Fits a Laplace density to the normed histogram of each latent variable feature map.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variables. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of
        the ith example.
    path_to_idx_map_exception : str
        Path to the file which stores the index
        of the map that is not compressed as the
        other maps. The path must end with ".pkl".
    path_to_checking_f : str
        Path to the folder storing the
        saved normed histograms, the saved
        histogram of the Laplace density scales
        and the saved histogram of the Laplace
        density locations.
    
    """
    locations = []
    scales = []
    with open(path_to_idx_map_exception, 'rb') as file:
        idx_map_exception = pickle.load(file)
    for i in range(y_float32.shape[3]):
        map_float32 = y_float32[:, :, :, i]
        max_abs_map = numpy.ceil(numpy.amax(numpy.absolute(map_float32))).item()
        
        # The grid below contains 50 points
        # per unit interval.
        grid = numpy.linspace(-max_abs_map,
                              max_abs_map,
                              num=100*int(max_abs_map) + 1)
        
        # Let's assume that `map_float32` contains i.i.d samples
        # from an unknown probability density function. The two
        # equations below result from the minimization of the
        # Kullback-Lieber divergence of the unknown probability
        # density function from our statistical model (Laplace
        # density of location `laplace_location` and scale
        # `laplace_scale`). Note that this minimization is
        # equivalent to the maximum likelihood estimator.
        # To dive into the details, see:
        # "Estimating distributions and densities". 36-402,
        # advanced data analysis, CMU, 27 January 2011.
        laplace_location = numpy.mean(map_float32).item()
        laplace_scale = numpy.mean(numpy.absolute(map_float32 - laplace_location)).item()
        laplace_pdf = scipy.stats.laplace.pdf(grid,
                                              loc=laplace_location,
                                              scale=laplace_scale)
        handle = [plt.plot(grid, laplace_pdf, color='red')[0]]
        hist, bin_edges = numpy.histogram(map_float32,
                                          bins=60,
                                          density=True)
        plt.bar(bin_edges[0:60],
                hist,
                width=bin_edges[1] - bin_edges[0],
                align='edge',
                color='blue')
        plt.title('Latent variable feature map {}'.format(i + 1))
        plt.legend(handle,
                   [r'$f( . ; {0}, {1})$'.format(str(round(laplace_location, 2)), str(round(laplace_scale, 2)))],
                   prop={'size': 30},
                   loc=9)
        plt.savefig(os.path.join(path_to_checking_f, 'fitting_map_{}.png'.format(i + 1)))
        plt.clf()
        if i != idx_map_exception:
            locations.append(laplace_location)
            scales.append(laplace_scale)
    
    # `nb_kept` must be equal to `y_float32.shape[3] - 1`.
    nb_kept = len(locations)
    tls.histogram(numpy.array(locations),
                  'Histogram of {} locations'.format(nb_kept),
                  os.path.join(path_to_checking_f, 'laplace_locations.png'))
    tls.histogram(numpy.array(scales),
                  'Histogram of {} scales'.format(nb_kept),
                  os.path.join(path_to_checking_f, 'laplace_scales.png'))

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
    if not os.path.exists(path_to_checking_f):
        os.makedirs(path_to_checking_f)
    reference_uint8 = numpy.load('datasets/kodak/results/kodak.npy')
    (_, h_in, w_in) = reference_uint8.shape
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
    fitting_eae_kodak(y_float32,
                      path_to_idx_map_exception,
                      path_to_checking_f)


