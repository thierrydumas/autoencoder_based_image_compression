"""A script to fit a Laplace density to the normed histogram of the latent variables in a trained entropy autoencoder.

250 digits from the SVHN test set are used
for the fitting.

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

import parsing.parsing
import svhn.svhn
import tools.tools as tls

def fitting_eae_svhn(reference_float64, entropy_ae, title, path):
    """Fits a Laplace density to the normed histogram of the latent variables in the trained entropy autoencoder.
    
    Parameters
    ----------
    reference_float64 : numpy.ndarray
        2D array with data-type `numpy.float64`.
        RGB digits after the preprocessing.
        `reference_float64[i, :]` contains the
        ith RGB digit after the preprocessing.
    entropy_ae : EntropyAutoencoder
        Entropy autoencoder trained with a
        specific scaling coefficient.
    title : str
        Title of the saved normed histogram.
    path : str
        Path to the saved normed histogram. The
        path must end with ".png".
    
    """
    y = entropy_ae.encoder(reference_float64)[1]
    edge_left = numpy.floor(numpy.amin(y)).item()
    edge_right = numpy.ceil(numpy.amax(y)).item()
    
    # The grid below contains 50 points
    # per unit interval.
    grid = numpy.linspace(edge_left,
                          edge_right,
                          num=50*int(edge_right - edge_left) + 1)
    
    # Let's assume that `y` contains i.i.d samples from
    # an unknown probability density function. The two
    # equations below result from the minimization of
    # the Kullback-Lieber divergence of the unknown
    # probability density function from our statistical
    # model (Laplace density of location `laplace_location`
    # and scale `laplace_scale`). Note that this minimization
    # is equivalent to the maximum likelihood estimator.
    # To dive into the details, see:
    # "Estimating distributions and densities". 36-402,
    # advanced data analysis, CMU, 27 January 2011.
    laplace_location = numpy.mean(y).item()
    laplace_scale = numpy.mean(numpy.absolute(y - laplace_location)).item()
    laplace_pdf = scipy.stats.laplace.pdf(grid,
                                          loc=laplace_location,
                                          scale=laplace_scale)
    handle = [plt.plot(grid, laplace_pdf, color='red')[0]]
    hist, bin_edges = numpy.histogram(y,
                                      bins=60,
                                      density=True)
    plt.bar(bin_edges[0:60],
            hist,
            width=bin_edges[1] - bin_edges[0],
            align='edge',
            color='blue')
    plt.title(title)
    plt.legend(handle,
               [r'$f( . ; {0}, {1})$'.format(str(round(laplace_location, 2)), str(round(laplace_scale, 2)))],
               prop={'size': 30},
               loc=9)
    plt.savefig(path)
    plt.clf()

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
    fitting_eae_svhn(reference_float64,
                     entropy_ae,
                     'Latent variables',
                     os.path.join(path_to_checking_f, 'fitting_laplace.png'))


