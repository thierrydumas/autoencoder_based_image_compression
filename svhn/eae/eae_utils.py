"""A library that contains functions dedicated to the entropy autoencoder."""

import numpy

import svhn.svhn
import tools.tools as tls

def compute_rate_psnr(reference_uint8, mean_training, std_training, entropy_ae,
                      bin_width, nb_vertically, path_to_reconstruction):
    """Computes the mean rate and the mean PNSR associated to the compression of the RGB digits via an entropy autoencoder.
    
    An image of the RGB digits after being compressed
    via the entropy autoencoder is saved.
    
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
    entropy_ae : EntropyAutoencoder
        Entropy autoencoder trained with a
        specific scaling coefficient.
    bin_width : float
        Quantization bin width.
    nb_vertically : int
        Number of RGB digits per column in
        the saved image.
    path_to_reconstruction : str
        Path to the saved image of the RGB digits
        after being compressed via the entropy
        autoencoder. The path must end with ".png".
    
    Returns
    -------
    tuple
        numpy.float64
            Mean rate associated to the compression of
            the RGB digits via the entropy autoencoder.
        numpy.float64
            Mean PSNR associated to the compression of
            the RGB digits via the entropy autoencoder.
    
    """
    # The function `svhn.svhn.preprocess_svhn` checks
    # that `reference_uint8.dtype` is equal to `numpy.uint8`
    # and `reference_uint8.ndim` is equal to 2.
    reference_float64 = svhn.svhn.preprocess_svhn(reference_uint8,
                                                  mean_training,
                                                  std_training)
    (nb_images, nb_pixels) = reference_uint8.shape
    
    # At training time, the decoder was fed with
    # latent variables perturbed by uniform noise.
    # However, at test time, the decoder is fed with
    # quantized latent variables.
    y = entropy_ae.encoder(reference_float64)[1]
    quantized_y = tls.quantization(y, bin_width)
    
    # In the function `tls.discrete_entropy`, `quantized_y`
    # is flattened to compute the entropy.
    disc_entropy = tls.discrete_entropy(quantized_y, bin_width)
    rate = entropy_ae.nb_y*disc_entropy/nb_pixels
    reconstruction_float64 = entropy_ae.decoder(quantized_y)[1]
    rec_rescaled_float64 = reconstruction_float64*std_training + \
        numpy.tile(mean_training, (nb_images, 1))
    reconstruction_uint8 = tls.cast_float_to_uint8(rec_rescaled_float64)
    psnr = tls.mean_psnr(reference_uint8, reconstruction_uint8)
    tls.visualize_rows(reconstruction_uint8,
                       32,
                       32,
                       nb_vertically,
                       path_to_reconstruction)
    return (rate, psnr)

def preliminary_fitting(training_uint8, mean_training, std_training, entropy_ae,
                        batch_size, nb_epochs_fitting):
    """"Pre-trains the parameters of the piecewise linear function.
    
    Parameters
    ----------
    training_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Training set. `training_uint8[i, :]`
        contains the ith training image.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each pixel
        over all training images.
    entropy_ae : EntropyAutoencoder
        Entropy auto-encoder trained with a
        specific scaling coefficient.
    batch_size : int
        Size of the mini-batches.
    nb_epochs_fitting : int
        Number of fitting epochs.
    
    """
    nb_batches = tls.subdivide_set(training_uint8.shape[0], batch_size)
    for _ in range(nb_epochs_fitting):
        for j in range(nb_batches):
            batch_uint8 = training_uint8[j*batch_size:(j + 1)*batch_size, :]
            
            # The function `svhn.svhn.preprocess_svhn` checks
            # that `batch_uint8.dtype` is equal to `numpy.uint8`
            # and `batch_uint8.ndim` is equal to 2.
            batch_float64 = svhn.svhn.preprocess_svhn(batch_uint8,
                                                      mean_training,
                                                      std_training)
            entropy_ae.training_fct(batch_float64)


