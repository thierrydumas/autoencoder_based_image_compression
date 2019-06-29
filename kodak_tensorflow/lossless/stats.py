"""A library that contains functions for collecting statistics on the latent variable feature maps in an entropy autoencoder."""

import numpy
import os
import pickle

import eae.batching
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

def compute_probabilities_intervals(data, size_interval):
    """Computes the probability that a data value belongs to an axis interval.
    
    Parameters
    ----------
    data : numpy.ndarray
        1D array.
        Data values.
    size_interval : float
        Size of the intervals in the axis.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Axis.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Probability that a data value belongs to an
            axis interval, for each axis interval.
    
    Raises
    ------
    ValueError
        If the interval size exceeds the range of the data values.
    ValueError
        If the range of the data values cannot be split into an
        integer number of intervals of size `size_interval`.
    
    """
    edge_left = numpy.floor(numpy.amin(data)).item()
    edge_right = numpy.ceil(numpy.amax(data)).item()
    difference_edges = edge_right - edge_left
    if difference_edges < size_interval:
        raise ValueError('The interval size exceeds the range of the data values.')
    nb_edges_minus_1_float = difference_edges/size_interval
    if nb_edges_minus_1_float.is_integer():
        nb_edges = int(nb_edges_minus_1_float) + 1
    else:
        raise ValueError('The range of the data values cannot be split into '
                         + 'an integer number of intervals of size {}.'.format(size_interval))
    
    # The left edge of the histogram is smaller than the
    # smallest element of `data`. The right edge of the
    # histogram is larger than the largest element of
    # `data`.
    bin_edges = numpy.linspace(edge_left,
                               edge_right,
                               num=nb_edges)
    
    # In the function `numpy.histogram`, `data`
    # is flattened to compute the histogram.
    hist = numpy.histogram(data,
                           bins=bin_edges,
                           density=True)[0]
    
    # The probability that a data value belongs to
    # [`bin_edges[i]`, `bin_edges[i + 1]`] is the integral
    # over [`bin_edges[i]`, `bin_edges[i + 1]`] of the estimated
    # probability density function of the data.
    # Warning! `hist[i]` is the value of the probability
    # density function of the data at the middle of
    # [`bin_edges[i]`, `bin_edges[i + 1]`].
    return (bin_edges, hist*size_interval)

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
    divergences = numpy.zeros(y_float32.shape[3])
    for i in range(y_float32.shape[3]):
        probs = compute_probabilities_intervals(y_float32[:, :, :, i], 1.)[1]
        probs_non_zero = numpy.extract(probs != 0.,
                                       probs)
        nb_remaining_probs = probs_non_zero.size
        
        # If a latent variable feature map contains
        # exclusively elements very close to 0.0,
        # `nb_remaining_probs` is equal to 1.
        if nb_remaining_probs > 1:
            uniform_probs = (1./nb_remaining_probs)*numpy.ones(nb_remaining_probs)
            divergences[i] = tls.jensen_shannon_divergence(probs_non_zero,
                                                           uniform_probs)
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
        The path ends with ".npy".
    path_to_idx_map_exception : str
        Path to the file in which the index
        of the map that is not compressed as
        the other maps is saved. The path
        ends with ".pkl".
    paths_to_binary_probabilities : list
        The ith string in this list is the path
        to the file in which the binary probabilities
        associated to the ith multiplier are saved.
        Each path ends with ".npy".
    
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
        
        # The function `eae.batching.encode_mini_batches` checks
        # that `luminances_uint8.dtype` is equal to `numpy.uint8`.
        y_float32 = eae.batching.encode_mini_batches(luminances_uint8,
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


