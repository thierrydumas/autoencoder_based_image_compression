"""A library that contains functions for analyzing the trained entropy autoencoders."""

import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import os
import scipy.stats

import eae.graph.constants as csts
import tools.tools as tls

def activate_latent_variable(sess, isolated_decoder, h_in, w_in, bin_widths, row_activation, col_activation,
                             idx_map_activation, activation_value, map_mean, height_crop, width_crop, path_to_crop):
    """Activates one latent variable and deactivates the others.
    
    One latent variable is activated and the others
    are deactivated. Then, the latent variable feature
    maps are quantized. Finally, the quantized latent
    variable feature maps are passed through the decoder
    of the entropy autoencoder.
    
    Parameters
    ----------
    sess : Session
        Session that runs the graph.
    isolated_decoder : IsolatedDecoder
        Decoder of the entropy autoencoder. The graph
        of the decoder is built to process one example
        at a time.
    h_in : int
        Height of the images returned by the
        isolated decoder.
    w_in : int
        Width of the images returned by the
        isolated decoder.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths at the end of the
        training.
    row_activation : int
        Row of the activated latent variable in the
        latent variable feature map of index `idx_map_activation`.
    col_activation : int
        Column of the activated latent variable in the
        latent variable feature map of index `idx_map_activation`.
    idx_map_activation : int
        Index of the latent variable feature map
        containing the activated latent variable.
    activation_value : float
        Activation value.
    map_mean : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Latent variable feature map means.
    height_crop : int
        Height of the crop of the decoder output.
    width_crop : int
        Width of the crop of the decoder output.
    path_to_crop : str
        Path to the saved crop of the decoder
        output. The path ends with ".png".
    
    Raises
    ------
    ValueError
        If `row_activation` is not strictly positive.
    ValueError
        If `col_activation` is not strictly positive.
    
    """
    if row_activation <= 0:
        raise ValueError('`row_activation` is not strictly positive.')
    if col_activation <= 0:
        raise ValueError('`col_activation` is not strictly positive.')
    y_float32 = numpy.tile(numpy.reshape(map_mean, (1, 1, 1, csts.NB_MAPS_3)),
                           (1, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, 1))
    y_float32[0, row_activation, col_activation, idx_map_activation] = activation_value
    quantized_y_float32 = tls.quantize_per_map(y_float32, bin_widths)
    reconstruction_float32 = sess.run(
        isolated_decoder.node_reconstruction,
        feed_dict={isolated_decoder.node_quantized_y:quantized_y_float32}
    )
    reconstruction_uint8 = numpy.squeeze(tls.cast_bt601(reconstruction_float32), axis=(0, 3))
    
    # The ratio between the feature map height and the
    # decoded image height is equal to `csts.STRIDE_PROD`.
    row_offset = (row_activation - 1)*csts.STRIDE_PROD
    col_offset = (col_activation - 1)*csts.STRIDE_PROD
    tls.save_image(path_to_crop,
                   reconstruction_uint8[row_offset:row_offset + height_crop, col_offset:col_offset + width_crop])

def fit_maps(y_float32, idx_map_exception, path_to_histogram_locations, path_to_histogram_scales, paths):
    """Fits a Laplace density to the normed histogram of each latent variable feature map.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variables. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of
        the ith example.
    idx_map_exception : int
        Index of the latent variable feature map
        that is not compressed as the other maps.
    path_to_histogram_locations : str
        Path to the histogram of the Laplace locations. The
        path ends with ".png".
    path_to_histogram_scales : str
        Path to the histogram of the Laplace scales. The
        path ends with ".png".
    paths : list
        `paths[i]` is the path to the fitted normed histogram
        for the ith latent variable feature map. Each path ends
        with ".png".
    
    Raises
    ------
    ValueError
        If `len(paths)` is not equal to `y_float32.shape[3]`.
    
    """
    if len(paths) != y_float32.shape[3]:
        raise ValueError('`len(paths)` is not equal to `y_float32.shape[3]`.')
    locations = []
    scales = []
    for i in range(y_float32.shape[3]):
        map_float32 = y_float32[:, :, :, i]
        edge_left = numpy.floor(numpy.amin(map_float32)).item()
        edge_right = numpy.ceil(numpy.amax(map_float32)).item()
        
        # The grid below contains 50 points
        # per unit interval.
        grid = numpy.linspace(edge_left,
                              edge_right,
                              num=50*int(edge_right - edge_left) + 1)
        
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
        plt.savefig(paths[i])
        plt.clf()
        if i != idx_map_exception:
            locations.append(laplace_location)
            scales.append(laplace_scale)
    
    # `nb_kept` must be equal to `y_float32.shape[3] - 1`.
    nb_kept = len(locations)
    tls.histogram(numpy.array(locations),
                  'Histogram of {} locations'.format(nb_kept),
                  path_to_histogram_locations)
    tls.histogram(numpy.array(scales),
                  'Histogram of {} scales'.format(nb_kept),
                  path_to_histogram_scales)

def mask_maps(y_float32, sess, isolated_decoder, bin_widths, idx_unmasked_map, map_mean, height_crop, width_crop, paths):
    """Masks all the latent variable feature maps except one.
    
    All the latent variable feature maps except one
    are masked. Then, the latent variable feature maps
    are quantized. Finally, the quantized latent
    variable feature maps are passed through the
    decoder of the entropy autoencoder.
    
    Parameters
    ----------
    y_float32 : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Latent variables. `y_float32[i, :, :, j]`
        is the jth latent variable feature map of
        the ith example.
    sess : Session
        Session that runs the graph.
    isolated_decoder : IsolatedDecoder
        Decoder of the entropy autoencoder. The graph
        of the decoder is built to process one example
        at a time.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths at the end of the
        training.
    idx_unmasked_map : int
        Index of the unmasked latent variable
        feature map.
    map_mean : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Latent variable feature map means.
    height_crop : int
        Height of the crop of the decoder output.
    width_crop : int
        Width of the crop of the decoder output.
    paths : list
        The ith string in this list is the path
        to the ith saved crop of the decoder output.
        Each path ends with ".png".
    
    Raises
    ------
    ValueError
        If `len(paths)` is not equal to `y_float32.shape[0]`.
    
    """
    if len(paths) != y_float32.shape[0]:
        raise ValueError('`len(paths)` is not equal to `y_float32.shape[0]`.')
        
    # The same latent variable feature map is
    # iteratively overwritten in the loop below.
    masked_y_float32 = numpy.tile(numpy.reshape(map_mean, (1, 1, 1, y_float32.shape[3])),
                                  (1, y_float32.shape[1], y_float32.shape[2], 1))
    for i in range(y_float32.shape[0]):
        masked_y_float32[0, :, :, idx_unmasked_map] = y_float32[i, :, :, idx_unmasked_map]
        quantized_y_float32 = tls.quantize_per_map(masked_y_float32, bin_widths)
        reconstruction_float32 = sess.run(
            isolated_decoder.node_reconstruction,
            feed_dict={isolated_decoder.node_quantized_y:quantized_y_float32}
        )
        reconstruction_uint8 = numpy.squeeze(tls.cast_bt601(reconstruction_float32),
                                             axis=(0, 3))
        tls.save_image(paths[i],
                       reconstruction_uint8[0:height_crop, 0:width_crop])


