"""A library that contains functions for collecting statistics on the latent variable feature maps in an entropy autoencoder."""

import numpy
import os
import pickle

import eae.eae_utils as eaeuls
import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def compute_binary_probabilities(y_float32, bin_widths_test, map_mean, truncated_unary_length):
    """Computes the binary probabilities associated to each latent variable feature map.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variable feature maps. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of the ith
        example.
    bin_widths_test : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Test quantization bin widths.
    map_mean : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Latent variable feature map means.
    truncated_unary_length : int
        Length of the truncated unary prefix.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float64`.
        Binary probabilities. The element at the position
        [i, j] in this array is the probability the jth
        binary decision is 0 in the truncated unary prefix
        associated to the ith absolute centered-quantized
        latent variable feature map.
    
    """
    (nb_images, height_map, width_map, nb_maps) = y_float32.shape
    centered_y_float32 = y_float32 - numpy.tile(map_mean, (nb_images, height_map, width_map, 1))
    centered_quantized_y_float32 = tls.quantize_per_map(centered_y_float32, bin_widths_test)
    cumulated_zeros = numpy.zeros((nb_maps, truncated_unary_length), dtype=numpy.int64)
    cumulated_ones = numpy.zeros((nb_maps, truncated_unary_length), dtype=numpy.int64)
    for i in range(nb_maps):
        (cumulated_zeros[i, :], cumulated_ones[i, :]) = \
            count_binary_decisions(numpy.absolute(centered_quantized_y_float32[:, :, :, i]),
                                   bin_widths_test[i].item(),
                                   truncated_unary_length)
    total = cumulated_zeros + cumulated_ones
    
    # For the ith absolute centered-quantized latent
    # variable feature map, if the jth binary decision
    # never occurs, `binary_probabilities[i, j]`
    # is equal to `numpy.nan`.
    with numpy.errstate(invalid='ignore'):
        binary_probabilities = cumulated_zeros.astype(numpy.float64)/total.astype(numpy.float64)
    
    # If a binary decision never occurs, the
    # probability this binary decision is 0
    # is set to 0.5.
    binary_probabilities[numpy.isnan(binary_probabilities)] = 0.5
    binary_probabilities[binary_probabilities == 0.] = 0.01
    binary_probabilities[binary_probabilities == 1.] = 0.99
    return binary_probabilities

def count_binary_decisions(abs_centered_quantized_data, bin_width_test, truncated_unary_length):
    """Counts the number of occurrences of 0 for each binary decision in the truncated unary prefix of the absolute centered-quantized data.
    
    Parameters
    ----------
    abs_centered_quantized_data : numpy.ndarray
        Array with data-type `numpy.float32`.
        Absolute centered-quantized data.
    bin_width_test : float
        Test quantization bin width. Previously,
        a uniform scalar quantization with quantization
        bin width `bin_width_test` was applied to zero-mean
        data and the result was passed through the absolute
        function, giving rise to `abs_centered_quantized_data`.
    truncated_unary_length : int
        Length of the truncated unary prefix.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.int64`.
            Number of occurrences of 0 for each binary
            decision in the truncated unary prefix of
            the absolute centered-quantized data.
        numpy.ndarray
            1D array with data-type `numpy.int64`.
            Number of occurrences of 1 for each binary
            decision in the truncated unary prefix of
            the absolute centered-quantized data.
    
    Raises
    ------
    ValueError
        If an element of `abs_centered_quantized_data`
        is not positive.
    
    """
    if numpy.any(abs_centered_quantized_data < 0.):
        raise ValueError('An element of `abs_centered_quantized_data` is not positive.')
    
    # In the function `tls.count_symbols`, `abs_centered_quantized_data`
    # is flattened beforehand.
    hist = tls.count_symbols(abs_centered_quantized_data,
                             bin_width_test)
    cumulated_zeros = numpy.zeros(truncated_unary_length, dtype=numpy.int64)
    cumulated_ones = numpy.zeros(truncated_unary_length, dtype=numpy.int64)
    
    # Even though the data was centered before being quantized,
    # the smallest symbol in `abs_centered_quantized_data` is
    # not necessarily 0.0.
    minimum = int(round(numpy.amin(abs_centered_quantized_data).item()/bin_width_test))
    for i in range(hist.size):
        ii = i + minimum
        if ii < truncated_unary_length:
            cumulated_ones[0:ii] += hist[i]
            cumulated_zeros[ii] += hist[i]
        else:
            cumulated_ones += hist[i]
    return (cumulated_zeros, cumulated_ones)

def find_index_map_exception(y_float32):
    """Finds the index of the latent variable feature map that is not compressed as the other maps.
    
    The code is based on an observation. For all
    latent variable feature maps, except one, the
    distribution is close to a Laplace distribution.
    The distribution of the exception is relatively
    close to the uniform distribution. To find the
    index of the exception, we look for the latent
    variable feature map that minimizes the Shannon-Jensen
    divergence between its distribution and the uniform
    distribution.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variable feature maps. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of the ith
        example.
    
    Returns
    -------
    int
        Index of the latent variable feature map
        that is not compressed as the other maps.
    
    """
    nb_maps = y_float32.shape[3]
    divergences = numpy.zeros(nb_maps)
    for i in range(nb_maps):
        map_float32 = y_float32[:, :, :, i]
        middle_1st_bin = numpy.round(numpy.amin(map_float32)).item()
        middle_last_bin = numpy.round(numpy.amax(map_float32)).item()
        nb_edges = int(middle_last_bin - middle_1st_bin) + 2
        
        # The 1st edge of the histogram is smaller than the
        # smallest element of `map_float32`. The last edge
        # of the histogram is larger than the largest element
        # of `map_float32`.
        bin_edges = numpy.linspace(middle_1st_bin - 0.5,
                                   middle_last_bin + 0.5,
                                   num=nb_edges)
        
        # In the function `numpy.histogram`, `map_float32`
        # is flattened to compute the histogram.
        hist = numpy.histogram(map_float32,
                               bins=bin_edges,
                               density=True)[0]
        hist_non_zero = numpy.extract(hist != 0., hist)
        nb_remaining_bins = hist_non_zero.size
        
        # If a latent variable feature map contains
        # exclusively elements very close to 0.0,
        # `nb_remaining_bins` is equal to 1.
        if nb_remaining_bins > 1:
            uniform_probs = (1./nb_remaining_bins)*numpy.ones(nb_remaining_bins)
            divergences[i] = tls.jensen_shannon_divergence(hist_non_zero, uniform_probs)
        else:
            divergences[i] = 1.
    return numpy.argmin(divergences).item()

def save_statistics(luminances_uint8, sess, entropy_ae, batch_size, multipliers, truncated_unary_length,
                    path_to_map_mean, path_to_idx_map_exception, paths_to_binary_probabilities):
    """Saves some statistics on the latent variable feature maps in the entropy autoencoder.
    
    Parameters
    ----------
    luminances_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images. `luminances_uint8[i, :, :, :]`
        is the ith luminance image. The last dimension of
        `luminances_uint8` is equal to 1.
    sess : Session
        Session that runs the graph.
    entropy_ae : EntropyAutoencoder
        Entropy auto-encoder trained with a specific
        scaling coefficient.
    batch_size : int
        Size of the mini-batches.
    multipliers : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Multipliers. All the quantization bin widths
        at the end of the training are multiplied
        by the ith multiplier, providing the ith
        set of test quantization bin widths.
    truncated_unary_length : int
        Length of the truncated unary prefix.
    path_to_map_mean : str
        Path to the file in which the latent
        variable feature map means are saved.
        The path must end with ".npy".
    path_to_idx_map_exception : str
        Path to the file in which the index
        of the map that is not compressed as
        the other maps is saved. The path
        must end with ".pkl".
    paths_to_binary_probabilities : list
        The ith string in this list is the path
        to the file in which the binary probabilities
        associated to the ith multiplier are saved.
        Each path must end with ".npy".
    
    Raises
    ------
    ValueError
        If `len(paths_to_binary_probabilities)` is not equal
        to `multipliers.size`.
    
    """
    nb_multipliers = multipliers.size
    if len(paths_to_binary_probabilities) != nb_multipliers:
        raise ValueError('`len(paths_to_binary_probabilities)` is not equal to `multipliers.size`.')
    booleans = [os.path.isfile(path_to_binary_probability) for path_to_binary_probability in paths_to_binary_probabilities]
    if os.path.isfile(path_to_map_mean) and os.path.isfile(path_to_idx_map_exception) and all(booleans):
        print('The statistics on the latent variable feature maps already exist.')
        print('Delete them manually to recompute them.')
    else:
        
        # The function `eaeuls.encode_mini_batches` checks
        # that `luminances_uint8.dtype` is equal to `numpy.uint8`.
        y_float32 = eaeuls.encode_mini_batches(luminances_uint8,
                                               sess,
                                               entropy_ae,
                                               batch_size)
        map_mean = numpy.mean(y_float32, axis=(0, 1, 2))
        numpy.save(path_to_map_mean, map_mean)
        idx_map_exception = find_index_map_exception(y_float32)
        with open(path_to_idx_map_exception, 'wb') as file:
            pickle.dump(idx_map_exception, file, protocol=2)
        for i in range(nb_multipliers):
            
            # The method `get_bin_widths` of class `EntropyAutoencoder`
            # returns the quantization bin widths at the end of the training.
            bin_widths_test = multipliers[i]*entropy_ae.get_bin_widths()
            binary_probabilities = compute_binary_probabilities(y_float32,
                                                                bin_widths_test,
                                                                map_mean,
                                                                truncated_unary_length)
            numpy.save(paths_to_binary_probabilities[i], binary_probabilities)


