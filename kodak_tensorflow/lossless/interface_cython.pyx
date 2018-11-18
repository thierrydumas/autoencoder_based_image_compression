"""A library that defines a Cython interface which runs the C++ lossless coder."""

import numpy
cimport numpy

cdef extern from "c++/source/compression.h":
    cdef numpy.uint32_t compress_lossless(const numpy.uint32_t&,
                                          const numpy.int16_t* const,
                                          numpy.int16_t* const,
                                          const numpy.uint8_t&,
                                          const double* const) except +

def compress_lossless_flattened_map(numpy.ndarray[numpy.int16_t, ndim=1] ref_map_int16,
                                    numpy.ndarray[double, ndim=1] probabilities):
    """Compresses without loss a flattened map of signed integers.
    
    Parameters
    ----------
    ref_map_int16 : numpy.ndarray
        1D array with data-type `numpy.int16`.
        Flattened map of signed integers to be
        compressed without loss.
    probabilities : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Its ith element is the probability that the
        ith binary decision is 0 in the truncated
        unary prefix. The number of elements in
        `probabilities` cannot be larger than 256.
        This means that the truncated unary prefix
        cannot be larger than 256.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.int16`.
            Reconstruction of the flattened map of
            signed integers after the compression
            without loss.
        int
            Coding cost of the flattened map of
            signed integers. The coding cost is
            expressed in bits.
    
    """
    # Cython automatically checks the number of dimensions
    # and the data-type of `ref_map_int16` and `probabilities`.
    cdef numpy.uint32_t size = ref_map_int16.size
    
    # If `probabilities.size` does not belong to
    # [|0, 255|], Cython raises an exception.
    cdef numpy.uint8_t truncated_unary_length = probabilities.size
    cdef numpy.ndarray[numpy.int16_t, ndim=1] rec_map_int16 = numpy.zeros(size, dtype=numpy.int16)
    cdef numpy.uint32_t nb_bits = compress_lossless(size,
                                                    &ref_map_int16[0],
                                                    &rec_map_int16[0],
                                                    truncated_unary_length,
                                                    &probabilities[0])
    return (rec_map_int16, nb_bits)


