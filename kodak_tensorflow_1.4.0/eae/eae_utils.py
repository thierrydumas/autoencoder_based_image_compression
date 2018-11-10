"""A library that contains functions dedicated to the entropy autoencoder."""

import numpy

import eae.graph.constants as csts
import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def encode_mini_batches(luminances_uint8, sess, entropy_ae, batch_size):
    """Computes the latent variables from the luminance images via the entropy autoencoder, one mini-batch at a time.
    
    Parameters
    ----------
    luminances_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images. `luminances_uint8[i, :, :, :]`
        is the ith luminance image. The last dimension
        of `luminances_uint8` is equal to 1.
    sess : Session
        Session that runs the graph.
    entropy_ae : EntropyAutoencoder
        Entropy auto-encoder trained with a specific
        scaling coefficient.
    batch_size : int
        Size of the mini-batches.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variables.
    
    Raises
    ------
    AssertionError
        If `luminances_uint8.dtype` is not equal to `numpy.uint8`.
    AssertionError
        If `luminances_uint8.ndim` is not equal to 4.
    
    """
    assert luminances_uint8.dtype == numpy.uint8, '`luminances_uint8.dtype` is not equal to `numpy.uint8`.'
    assert luminances_uint8.ndim == 4, '`luminances_uint8.ndim` is not equal to 4.'
    (nb_images, h_in, w_in, _) = luminances_uint8.shape
    nb_batches = tls.subdivide_set(nb_images, batch_size)
    y_float32 = numpy.zeros((nb_images, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, csts.NB_MAPS_3), dtype=numpy.float32)
    for i in range(nb_batches):
        batch_float32 = luminances_uint8[i*batch_size:(i + 1)*batch_size, :, :, :].astype(numpy.float32)
        y_float32[i*batch_size:(i + 1)*batch_size, :, :, :] = sess.run(
            entropy_ae.node_y,
            feed_dict={entropy_ae.node_visible_units:batch_float32}
        )
    return y_float32

def preliminary_fitting(training_uint8, sess, entropy_ae, batch_size, nb_epochs_fitting):
    """"Pre-trains the parameters of the piecewise linear functions.
    
    Parameters
    ----------
    training_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Training set. `training_uint8[i, :, :, :]` is
        the ith training luminance image. The last
        dimension of `training_uint8` is equal to 1.
    sess : Session
        Session that runs the graph.
    entropy_ae : EntropyAutoencoder
        Entropy auto-encoder trained with a specific
        scaling coefficient.
    batch_size : int
        Size of the mini-batches.
    nb_epochs_fitting : int
        Number of fitting epochs.
    
    """
    nb_batches = tls.subdivide_set(training_uint8.shape[0], batch_size)
    for _ in range(nb_epochs_fitting):
        for j in range(nb_batches):
            batch_float32 = training_uint8[j*batch_size:(j + 1)*batch_size, :, :, :].astype(numpy.float32)
            entropy_ae.training_fct(sess, batch_float32)

def run_epoch_training(training_uint8, sess, entropy_ae, batch_size, nb_batches):
    """Trains the parameters the piecewise linear functions and the parameters of the entropy autoencoder for one epoch.
    
    Parameters
    ----------
    training_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Training set. `training_uint8[i, :, :, :]` is
        the ith training luminance image. The last
        dimension of `training_uint8` is equal to 1.
    sess : Session
        Session that runs the graph.
    entropy_ae : EntropyAutoencoder
        Entropy auto-encoder trained with a specific
        scaling coefficient.
    batch_size : int
        Size of the mini-batches.
    nb_batches : int
        Number of mini-batches scanned during
        the epoch.
    
    """
    permutation = numpy.random.permutation(training_uint8.shape[0])
    for i in range(nb_batches):
        batch_float32 = training_uint8[permutation[i*batch_size:(i + 1)*batch_size], :, :, :].astype(numpy.float32)
        
        # The parameters of the piecewise linear functions
        # and the parameters of the entropy autoencoder
        # are optimized alternatively.
        # For a given training batch, the method
        # `training_fct` of class `EntropyAutoencoder`
        # is called before the method `training_eae_bw`
        # as the first mentioned method verifies
        # whether the condition of expansion is met.
        entropy_ae.training_fct(sess, batch_float32)
        entropy_ae.training_eae_bw(sess, batch_float32)


