"""A library that contains functions for compressing via the C++ lossless coder."""

import numpy

import lossless.interface_cython
import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def compress_lossless_maps(ref_int16, path_to_binary_probabilities, idx_map_exception=-1):
    """Compresses without loss each map of signed integers separately.
    
    Parameters
    ----------
    ref_int16 : numpy.ndarray
        3D array with data-type `numpy.int16`.
        Stack of maps. `ref_int16[:, :, i]`
        is the ith map of signed integers to
        be compressed without loss.
    path_to_binary_probabilities : str
        Path to the file storing the binary
        probabilities. The path must end with ".npy".
    idx_map_exception : int, optional
        Index of the map that is not compressed
        as the other maps. The default value is -1,
        meaning that there is no exception: all
        the maps are compressed the same way.
    
    Returns
    -------
    tuple
        numpy.ndarray
            3D array with data-type `numpy.int16`.
            Reconstruction of the stack of maps after
            the compression without loss.
        numpy.ndarray
            1D array with data-type `numpy.uint32`.
            Coding costs. Its ith element is the coding
            cost of the ith map of signed integers. The
            coding cost is expressed in bits.
    
    Raises
    ------
    TypeError
        If `ref_int16.dtype` is not equal to `numpy.int16`.
    ValueError
        If `binary_probabilities.ndim` is not equal to 2.
    ValueError
        If `binary_probabilities.shape[0]` is not
        equal to `ref_int16.shape[2]`.
    
    """
    if ref_int16.dtype != numpy.int16:
        raise TypeError('`ref_int16.dtype` is not equal to `numpy.int16`.')
    (height_map, width_map, nb_maps) = ref_int16.shape
    binary_probabilities = numpy.load(path_to_binary_probabilities)
    if binary_probabilities.ndim != 2:
        raise ValueError('`binary_probabilities.ndim` is not equal to 2.')
    if binary_probabilities.shape[0] != nb_maps:
        raise ValueError('`binary_probabilities.shape[0]` is not equal to `ref_int16.shape[2]`.')
    rec_int16 = numpy.zeros((height_map, width_map, nb_maps), dtype=numpy.int16)
    nb_bits_each_map = numpy.zeros(nb_maps, dtype=numpy.uint32)
    for i in range(nb_maps):
        if i == idx_map_exception:
            
            # TODO: for the map that is not compressed as the other
            # maps, replace the approximation with the binary arithmetic
            # coder (even though the approximation is very good).
            cumulated_entropy = height_map*width_map*tls.discrete_entropy(ref_int16[:, :, i].astype(numpy.float32), 1.)
            nb_bits_each_map[i] = numpy.ceil(cumulated_entropy).astype(numpy.uint32)
            rec_int16[:, :, i] = ref_int16[:, :, i]
        else:
            ref_map_int16 = ref_int16[:, :, i].flatten()
            (rec_map_int16, nb_bits_each_map[i]) = \
                lossless.interface_cython.compress_lossless_flattened_map(ref_map_int16,
                                                                          binary_probabilities[i, :])
            rec_int16[:, :, i] = numpy.reshape(rec_map_int16, (height_map, width_map))
    return (rec_int16, nb_bits_each_map)

def rescale_compress_lossless_maps(centered_quantized_data, bin_widths_test, path_to_binary_probabilities, idx_map_exception=-1):
    """Rescales and compresses without loss each map of centered-quantized data separately.
    
    Parameters
    ----------
    centered_quantized_data : numpy.ndarray
        3D array with data-type `numpy.float32`.
        Centered-quantized data. `centered_quantized_data[:, :, i]`
        is the ith map of centered-quantized data to be rescaled
        and compressed without loss.
    bin_widths_test : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Test quantization bin widths. Previously,
        a uniform scalar quantization with quantization
        bin width `bin_widths_test[i]` was applied to zero-mean
        data, giving rise to `centered_quantized_data[:, :, i]`.
    path_to_binary_probabilities : str
        Path to the file storing the binary
        probabilities. The path must end with ".npy".
    idx_map_exception : int, optional
        Index of the map that is not compressed
        as the other maps. The default value is -1,
        meaning that there is no exception: all
        the maps are compressed the same way.
    
    Returns
    -------
    int
        Number of bits in the bitstream.
    
    Raises
    ------
    ValueError
        If `bin_widths_test.ndim` is not equal to 1.
    ValueError
        If `bin_widths_test.size` is not equal to
        `centered_quantized_data.shape[2]`.
    AssertionError
        If the lossless compression has altered
        the centered quantized data.
    
    """
    if bin_widths_test.ndim != 1:
        raise ValueError('`bin_widths_test.ndim` is not equal to 1.')
    
    # If `centered_quantized_data.ndim` is not equal to 3,
    # the unpacking below raises a `ValueError` exception.
    (height_map, width_map, nb_maps) = centered_quantized_data.shape
    if bin_widths_test.size != nb_maps:
        raise ValueError('`bin_widths_test.size` is not equal to `centered_quantized_data.shape[2]`.')
    tiled_bin_widths = numpy.tile(numpy.reshape(bin_widths_test, (1, 1, nb_maps)),
                                  (height_map, width_map, 1))
    
    # At the beginning of the function
    # `tls.cast_float_to_int16`, there is
    # a rounding operation. It corrects the
    # floating-point precision errors coming
    # from the division.
    ref_int16 = tls.cast_float_to_int16(centered_quantized_data/tiled_bin_widths)
    (rec_int16, nb_bits_each_map) = compress_lossless_maps(ref_int16,
                                                           path_to_binary_probabilities,
                                                           idx_map_exception=idx_map_exception)
    reconstruction = rec_int16.astype(numpy.float32)*tiled_bin_widths
    
    # Instead of returning `reconstruction`, we make
    # sure that `centered_quantized_data` and `reconstruction`
    # are identical.
    numpy.testing.assert_equal(centered_quantized_data,
                               reconstruction,
                               err_msg='The lossless compression has altered the centered quantized data.')
    return numpy.sum(nb_bits_each_map).item()


