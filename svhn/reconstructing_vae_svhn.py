"""A script to assess a trained variational autoencoder in terms of dimensionality reduction.

The trained variational autoencoder is assessed
on 250 RGB digits from the SVHN test set.

"""

import argparse
import numpy
import os
import pickle

import svhn.svhn
import tools.tools as tls

def compute_psnr(reference_uint8, mean_training, std_training, variational_ae, path_to_reconstruction):
    """Computes the mean PSNR associated to the dimensionality reduction of the RGB digits via a trained variational autoencoder.
    
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
    variational_ae : VariationalAutoencoder
        Trained variational autoencoder.
    path_to_reconstruction : str
        Path to the saved image of the RGB digits
        after reducing the dimensionality via the
        variational autoencoder. The path must end
        with ".png".
    
    Returns
    -------
    numpy.float64
        Mean PSNR associated to the dimensionality
        reduction of the RGB digits via the
        variational autoencoder.
    
    """
    # The function `svhn.svhn.preprocess_svhn` checks
    # that `reference_uint8.dtype` is equal to `numpy.uint8`
    # and `reference_uint8.ndim` is equal to 2.
    reference_float64 = svhn.svhn.preprocess_svhn(reference_uint8,
                                                  mean_training,
                                                  std_training)
    reconstruction_float64 = variational_ae.forward_pass(reference_float64)[5]
    rec_rescaled_float64 = reconstruction_float64*std_training + \
        numpy.tile(mean_training, (reference_uint8.shape[0], 1))
    reconstruction_uint8 = tls.cast_float_to_uint8(rec_rescaled_float64)
    psnr = tls.mean_psnr(reference_uint8, reconstruction_uint8)
    tls.visualize_rows(reconstruction_uint8,
                       32,
                       32,
                       10,
                       path_to_reconstruction)
    return psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assesses a trained variational autoencoder in terms of dimensionality reduction.')
    parser.parse_args()
    path_to_test = 'svhn/results/test_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    path_to_checking_r = 'vae/visualization/test/checking_reconstructing/'
    path_to_model = 'vae/results/vae_svhn.pkl'
    nb_images = 250
    with open(path_to_model, 'rb') as file:
        variational_ae = pickle.load(file)
    
    # `reference_uint8.dtype` is equal to `numpy.uint8`.
    reference_uint8 = numpy.load(path_to_test)[0:nb_images, :]
    tls.visualize_rows(reference_uint8,
                       32,
                       32,
                       10,
                       os.path.join(path_to_checking_r, 'reference.png'))
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    psnr = compute_psnr(reference_uint8,
                        mean_training,
                        std_training,
                        variational_ae,
                        os.path.join(path_to_checking_r, 'reconstruction.png'))
    print('Number of pixels: {}'.format(reference_uint8.shape[1]))
    print('Code length: {}'.format(variational_ae.nb_z))
    print('Mean PNSR over {0} RGB digits: {1}'.format(nb_images, psnr))


