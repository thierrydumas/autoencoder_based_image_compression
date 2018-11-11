"""A script to evaluate several trained entropy autoencoders, JPEG and JPEG2000 in terms of rate-distortion.

The trained entropy autoencoders, JPEG and JPEG2000
are evaluated on 250 RGB digits from the SVHN test set.

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

import eae.eae_utils as eaeuls
import jpeg.jpeg
import tools.tools as tls

def fix_gamma_fix_bin_width(reference_uint8, mean_training, std_training, bin_width_init,
                            multipliers, gamma, path_to_checking_r):
    """Computes a series of pairs (mean rate, mean PSNR).
    
    A single entropy autoencoder is considered.
    At training time, the quantization bin width
    was fixed.
    For each multiplier, the quantization bin width
    at training time is multiplied by the multiplier,
    yielding a test quantization bin width. Then, for
    each test quantization bin width, the pair
    (mean rate, mean PSNR) associated to the compression
    of the RGB digits via the single entropy autoencoder
    and the test quantization bin width is computed.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        RGB digits. `reference_uint8[i, :]`
        contains the ith RGB digit.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each pixel
        over all training images.
    bin_width_init : float
        Value of the quantization bin width at the
        beginning of the training.
    multipliers : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Multipliers.
    gamma : float
        Scaling coefficient of the single
        entropy autoencoder.
    path_to_checking_r : str
        Path to the folder containing the RGB
        digits after being compressed via
        entropy autoencoders.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean rate associated
            to the compression of the RGB digits via the
            single entropy autoencoder and the ith test
            quantization bin width.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean PSNR associated
            to the compression of the RGB digits via the
            single entropy autoencoder and the ith test
            quantization bin width.
    
    Raises
    ------
    AssertionError
        If the scaling coefficient written in the
        name of the file ".pkl" is incorrect.
    AssertionError
        If the quantization bin width written in the
        name of the file ".pkl" is incorrect.
    
    """
    suffix = '{0}_{1}'.format(tls.float_to_str(bin_width_init),
                              tls.float_to_str(gamma))
    path_to_storage = os.path.join(path_to_checking_r,
                                   'reconstruction_fix_gamma_fix_bin_width',
                                   suffix)
    path_to_model = 'eae/results/eae_svhn_{}.pkl'.format(suffix)
    with open(path_to_model, 'rb') as file:
        entropy_ae = pickle.load(file)
    assert entropy_ae.gamma == gamma, \
        'The file name is {0} whereas the scaling coefficient is {1}.'.format(path_to_model, entropy_ae.gamma)
    assert entropy_ae.bin_width == bin_width_init, \
        'The file name is {0} whereas the quantization bin width is {1}.'.format(path_to_model, entropy_ae.bin_width)
    nb_points = multipliers.size
    rate = numpy.zeros(nb_points)
    psnr = numpy.zeros(nb_points)
    for i in range(nb_points):    
        multiplier = multipliers[i].item()
        bin_width_test = multiplier*bin_width_init
        path_to_directory_reconstruction = os.path.join(path_to_storage,
                                                        'multiplier_{}'.format(tls.float_to_str(multiplier)))
        if not os.path.exists(path_to_directory_reconstruction):
            os.makedirs(path_to_directory_reconstruction)
        path_to_reconstruction = os.path.join(path_to_directory_reconstruction,
                                              'reconstruction.png')
        (rate[i], psnr[i]) = eaeuls.compute_rate_psnr(reference_uint8,
                                                      mean_training,
                                                      std_training,
                                                      entropy_ae,
                                                      bin_width_test,
                                                      10,
                                                      path_to_reconstruction)
    return (rate, psnr)

def vary_gamma_fix_bin_width(reference_uint8, mean_training, std_training, bin_width_init,
                             gammas, path_to_checking_r):
    """Computes a series of pairs (mean rate, mean PSNR).
    
    Several entropy autoencoders, each trained
    with a different scaling coefficient, are
    considered. At training time, the quantization
    bin width was fixed.
    For each scaling coefficient, the pair
    (mean rate, mean PSNR) associated to the compression
    of the RGB digits via the entropy autoencoder trained
    with the scaling coefficient is computed.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        RGB digits. `reference_uint8[i, :]`
        contains the ith RGB digit.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each pixel
        over all training images.
    bin_width_init : float
        Value of the quantization bin width at the
        beginning of the training. In this function,
        the quantization bin width is the same at
        training time and at test time.
    gammas : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Scaling coefficients.
    path_to_checking_r : str
        Path to the folder containing the RGB
        digits after being compressed via
        entropy autoencoders.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean rate associated
            to the compression of the RGB digits via the
            entropy autoencoder trained with the ith
            scaling coefficient.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean PSNR associated
            to the compression of the RGB digits via the
            entropy autoencoder trained with the ith
            scaling coefficient.
    
    Raises
    ------
    AssertionError
        If, for a file ".pkl", the scaling coefficient
        written in the file name is incorrect.
    AssertionError
        If, for a file ".pkl", the quantization bin width
        written in the file name is incorrect.
    
    """
    nb_points = gammas.size
    rate = numpy.zeros(nb_points)
    psnr = numpy.zeros(nb_points)
    for i in range(nb_points):
        suffix = '{0}_{1}'.format(tls.float_to_str(bin_width_init),
                                  tls.float_to_str(gammas[i].item()))
        path_to_model = 'eae/results/eae_svhn_{}.pkl'.format(suffix)
        with open(path_to_model, 'rb') as file:
            entropy_ae = pickle.load(file)
        assert entropy_ae.gamma == gammas[i].item(), \
            'The file name is {0} whereas the scaling coefficient is {1}.'.format(path_to_model, entropy_ae.gamma)
        assert entropy_ae.bin_width == bin_width_init, \
            'The file name is {0} whereas the quantization bin width is {1}.'.format(path_to_model, entropy_ae.bin_width)
        path_to_directory_reconstruction = os.path.join(path_to_checking_r,
                                                        'reconstruction_vary_gamma_fix_bin_width',
                                                        suffix)
        if not os.path.exists(path_to_directory_reconstruction):
            os.makedirs(path_to_directory_reconstruction)
        path_to_reconstruction = os.path.join(path_to_directory_reconstruction,
                                              'reconstruction.png')
        (rate[i], psnr[i]) = eaeuls.compute_rate_psnr(reference_uint8,
                                                      mean_training,
                                                      std_training,
                                                      entropy_ae,
                                                      bin_width_init,
                                                      10,
                                                      path_to_reconstruction)
    return (rate, psnr)

def vary_gamma_learn_bin_width(reference_uint8, mean_training, std_training, bin_width_init,
                               gammas, path_to_checking_r):
    """Computes a series of pairs (mean rate, mean PSNR).
    
    Several entropy autoencoders, each trained
    with a different scaling coefficient, are
    considered. At training time, the quantization
    bin width was learned.
    For each scaling coefficient, the pair
    (mean rate, mean PSNR) associated to the compression
    of the RGB digits via the entropy autoencoder trained
    with the scaling coefficient is computed.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        RGB digits. `reference_uint8[i, :]`
        contains the ith RGB digit.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each pixel
        over all training images.
    bin_width_init : float
        Value of the quantization bin width at the
        beginning of the training.
    gammas : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Scaling coefficients.
    path_to_checking_r : str
        Path to the folder containing the RGB
        digits after being compressed via
        entropy autoencoders.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean rate associated
            to the compression of the RGB digits via the
            entropy autoencoder trained with the ith
            scaling coefficient.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the mean PSNR associated
            to the compression of the RGB digits via the
            entropy autoencoder trained with the ith
            scaling coefficient.
    
    Raises
    ------
    AssertionError
        If, for a file ".pkl", the scaling coefficient
        written in the file name is incorrect.
    
    """
    nb_points = gammas.size
    rate = numpy.zeros(nb_points)
    psnr = numpy.zeros(nb_points)
    for i in range(nb_points):
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(bin_width_init),
                                              tls.float_to_str(gammas[i].item()))
        path_to_model = 'eae/results/eae_svhn_{}.pkl'.format(suffix)
        with open(path_to_model, 'rb') as file:
            entropy_ae = pickle.load(file)
        assert entropy_ae.gamma == gammas[i].item(), \
            'The file name is {0} whereas the scaling coefficient is {1}.'.format(path_to_model, entropy_ae.gamma)
        
        # The quantization bin width at the end
        # of the training is rounded to the 1st
        # digit after the decimal point, yielding
        # the test quantization bin width.
        bin_width_test = round(entropy_ae.bin_width, 1)
        path_to_directory_reconstruction = os.path.join(path_to_checking_r,
                                                        'reconstruction_vary_gamma_learn_bin_width',
                                                        suffix)
        if not os.path.exists(path_to_directory_reconstruction):
            os.makedirs(path_to_directory_reconstruction)
        path_to_reconstruction = os.path.join(path_to_directory_reconstruction,
                                              'reconstruction.png')
        (rate[i], psnr[i]) = eaeuls.compute_rate_psnr(reference_uint8,
                                                      mean_training,
                                                      std_training,
                                                      entropy_ae,
                                                      bin_width_test,
                                                      10,
                                                      path_to_reconstruction)
    return (rate, psnr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates several trained entropy autoencoders, JPEG and JPEG2000 in terms of rate-distortion.')
    parser.parse_args()
    dict_vary_gamma_fix_bin_width = {
        'bin_width_init': 1.,
        'gammas': numpy.array([5., 15., 45., 135.])
    }
    dict_vary_gamma_learn_bin_width = {
        'bin_width_init': 1.,
        'gammas': numpy.array([5., 15., 45., 135.])
    }
    dict_fix_gamma_fix_bin_width_0 = {
        'bin_width_init': 1.,
        'multipliers': numpy.array([0.1, 0.2, 0.5, 1., 2., 5., 10.]),
        'gamma': 15.
    }
    dict_fix_gamma_fix_bin_width_1 = {
        'bin_width_init': 0.2,
        'multipliers': numpy.array([0.5, 1., 2., 5., 10., 50.]),
        'gamma': 15.
    }
    path_to_test = 'svhn/results/test_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    path_to_checking_r = 'eae/visualization/test/checking_reconstructing/'
    nb_images = 250
    
    # The block below is dedicated to JPEG and
    # JPEG2000.
    path_to_before = 'jpeg/visualization/reference/'
    path_to_after = 'jpeg/visualization/reconstruction/'
    qualities_jpeg = [10, 20, 30, 40, 50, 60, 70, 80]
    qualities_jpeg2000 = [28, 30, 32, 34, 36, 38, 40, 42]
    
    # `reference_uint8.dtype` is equal to `numpy.uint8`.
    reference_uint8 = numpy.load(path_to_test)[0:nb_images, :]
    tls.visualize_rows(reference_uint8,
                       32,
                       32,
                       10,
                       path_to_checking_r + 'reference.png')
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    
    (rate_vary_gamma_fix_bin_width, psnr_vary_gamma_fix_bin_width) = \
        vary_gamma_fix_bin_width(reference_uint8,
                                 mean_training,
                                 std_training,
                                 dict_vary_gamma_fix_bin_width['bin_width_init'],
                                 dict_vary_gamma_fix_bin_width['gammas'],
                                 path_to_checking_r)
    (rate_vary_gamma_learn_bin_width, psnr_vary_gamma_learn_bin_width) = \
        vary_gamma_learn_bin_width(reference_uint8,
                                   mean_training,
                                   std_training,
                                   dict_vary_gamma_learn_bin_width['bin_width_init'],
                                   dict_vary_gamma_learn_bin_width['gammas'],
                                   path_to_checking_r)
    (rate_fix_gamma_fix_bin_width_0, psnr_fix_gamma_fix_bin_width_0) = \
        fix_gamma_fix_bin_width(reference_uint8,
                                mean_training,
                                std_training,
                                dict_fix_gamma_fix_bin_width_0['bin_width_init'],
                                dict_fix_gamma_fix_bin_width_0['multipliers'],
                                dict_fix_gamma_fix_bin_width_0['gamma'],
                                path_to_checking_r)
    (rate_fix_gamma_fix_bin_width_1, psnr_fix_gamma_fix_bin_width_1) = \
        fix_gamma_fix_bin_width(reference_uint8,
                                mean_training,
                                std_training,
                                dict_fix_gamma_fix_bin_width_1['bin_width_init'],
                                dict_fix_gamma_fix_bin_width_1['multipliers'],
                                dict_fix_gamma_fix_bin_width_1['gamma'],
                                path_to_checking_r)
    
    (rate_jpeg, psnr_jpeg, rate_jpeg2000, psnr_jpeg2000) = \
        jpeg.jpeg.evaluate_jpeg(reference_uint8,
                                path_to_before,
                                path_to_after,
                                qualities_jpeg,
                                qualities_jpeg2000)
    
    # The function `plt.plot` returns a list.
    handle = []
    handle.append(plt.plot(rate_vary_gamma_fix_bin_width,
                           psnr_vary_gamma_fix_bin_width,
                           color='orange',
                           marker='x',
                           markersize=9.)[0])
    handle.append(plt.plot(rate_vary_gamma_learn_bin_width,
                           psnr_vary_gamma_learn_bin_width,
                           color='green',
                           markerfacecolor='None',
                           marker='s',
                           markeredgecolor='green',
                           markersize=9.)[0])
    handle.append(plt.plot(rate_fix_gamma_fix_bin_width_0,
                           psnr_fix_gamma_fix_bin_width_0,
                           color='red',
                           markerfacecolor='None',
                           marker='o',
                           markeredgecolor='red',
                           markersize=9.)[0])
    handle.append(plt.plot(rate_fix_gamma_fix_bin_width_1,
                           psnr_fix_gamma_fix_bin_width_1,
                           color='yellow',
                           marker='+',
                           markersize=9.)[0])
    handle.append(plt.plot(rate_jpeg,
                           psnr_jpeg,
                           color='blue',
                           marker='<',
                           markersize=9.)[0])
    handle.append(plt.plot(rate_jpeg2000,
                           psnr_jpeg2000,
                           color='black',
                           marker='>',
                           markersize=9.)[0])
    plt.title('Mean rate-distortion curves over the first {} digits of the SVHN test set'.format(nb_images))
    plt.xlabel('mean rate (bbp)')
    plt.ylabel('mean PSNR (dB)')
    legend = [
        r'$\gamma$ varies, $\delta = {}$'.format(dict_vary_gamma_fix_bin_width['bin_width_init']),
        r'$\gamma$ varies, $\delta$ learned',
        r'$\gamma = {0}, \delta = {1}$'.format(dict_fix_gamma_fix_bin_width_0['gamma'],
                                               dict_fix_gamma_fix_bin_width_0['bin_width_init']),
        r'$\gamma = {0}, \delta = {1}$'.format(dict_fix_gamma_fix_bin_width_1['gamma'],
                                               dict_fix_gamma_fix_bin_width_1['bin_width_init']),
        'JPEG',
        'JPEG2000'
    ]
    plt.legend(handle,
               legend,
               loc='lower right',
               prop={'size':13},
               frameon=False)
    plt.savefig(path_to_checking_r + 'rate_distortion.png')
    plt.clf()


