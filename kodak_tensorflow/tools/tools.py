"""A library that contains common functions."""

# This library contains low level functions. The
# functions include many checks to avoid
# blunders when they are called.

import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy
import PIL.Image
import scipy.stats.distributions
import tarfile

# The functions are sorted in
# alphabetic order.

def average_entropies(data, bin_widths):
    """Quantizes the data and computes the mean entropy of the quantized data.
    
    Each map of data is quantized separately
    using a uniform scalar quantization.
    
    Parameters
    ----------
    data : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Data. `data[:, :, :, i]` is the ith map
        of data to be quantized.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths. `data[:, :, :, i]`
        is quantized using a uniform scalar quantization
        with quantization bin width `bin_widths[i]`.
    
    Returns
    -------
    numpy.float64
        Mean entropy of the quantized data.
    
    """
    quantized_data = quantize_per_map(data, bin_widths)
    nb_maps = data.shape[3]
    cumulated_entropy = 0.
    for i in range(nb_maps):
        
        # In the function `discrete_entropy`,
        # `quantized_data[:, :, :, i]` is flattened
        # to compute the entropy.
        cumulated_entropy += discrete_entropy(quantized_data[:, :, :, i],
                                              bin_widths[i].item())
    return cumulated_entropy/nb_maps

def cast_bt601(array_float):
    """Casts the array elements from float to 8-bit unsigned integer.
    
    The array elements correspond to
    luminance image pixels. Besides, the
    luminance images follow the ITU-R BT.601
    conversion. The array elements are clipped
    to [16., 235.], rounded to the nearest
    whole number and cast from float to 8-bit
    unsigned integer.
    
    Parameters
    ----------
    array_float : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
    
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.uint8`. It
        has the same shape as `array_float`.
    
    Raises
    ------
    TypeError
        If `array_float.dtype` is not smaller
        than `numpy.float` in type hierarchy.
    
    """
    if not numpy.issubdtype(array_float.dtype, numpy.float):
        raise TypeError('`array_float.dtype` is not smaller than `numpy.float` in type hierarchy.')
    return numpy.round(array_float.clip(min=16., max=235.)).astype(numpy.uint8)

def cast_float_to_int16(array_float):
    """Casts the array elements from float to 16-bit signed integer.
    
    The array elements are rounded to the nearest
    whole number and cast from float to 16-bit
    signed integer.
    
    Parameters
    ----------
    array_float : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
    
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.int16`. It
        has the same shape as `array_float`.
    
    Raises
    ------
    TypeError
        If `array_float.dtype` is not smaller
        than `numpy.float` in type hierarchy.
    AssertionError
        If the rounded array elements cannot be
        represented as 16-bit signed integers.
    
    """
    if not numpy.issubdtype(array_float.dtype, numpy.float):
        raise TypeError('`array_float.dtype` is not smaller than `numpy.float` in type hierarchy.')
    rounded_elements = numpy.round(array_float)
    
    # A 16-bit signed integer stores an integer
    # belonging to [|-32767, 32767|].
    numpy.testing.assert_array_less(numpy.absolute(rounded_elements),
                                    32768.,
                                    err_msg='The rounded array elements cannot be represented as 16-bit signed integers.')
    return rounded_elements.astype(numpy.int16)

def convert_approx_entropy(scaled_approx_entropy, gamma_scaling, nb_maps):
    """Converts the scaled cumulated approximate entropy of the quantized latent variables into its mean form.
    
    Parameters
    ----------
    scaled_approx_entropy : numpy.float64
        Scaled cumulated approximate entropy
        of the quantized latent variables.
    gamma_scaling : float
        Scaling coefficient. In the objective
        function to be minimized over the
        entropy autoencoder parameters, the
        scaling coefficient weights the cumulated
        approximate entropy of the quantized latent
        variables with respect to the reconstruction
        error and the l2-norm weight decay.
    nb_maps : int
        Number of latent variable feature maps.
    
    Returns
    -------
    numpy.float64
        Mean form of the scaled cumulated
        approximate entropy of the quantized
        latent variables.
    
    """
    return scaled_approx_entropy/(gamma_scaling*nb_maps)

def count_symbols(quantized_samples, bin_width):
    """Counts the number of occurrences in the quantized samples of each symbol.
    
    The symbols are ordered from the smallest to the
    largest. The 1st symbol is the smallest symbol. It
    is equal to the smallest quantized sample. The last
    symbol is the largest symbol. It is equal to the
    largest quantized sample. The difference between two
    successive symbols is the quantization bin width.
    
    Parameters
    ----------
    quantized_samples : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
        Quantized samples.
    bin_width : float
        Quantization bin width. Previously, samples
        were quantized using a uniform scalar quantization
        with quantization bin width `bin_width`, giving
        rise to `quantized_samples`.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.int64`.
        Number of occurrences in the quantized samples
        of each symbol.
    
    Raises
    ------
    ValueError
        If the quantization bin width is not
        strictly positive.
    AssertionError
        If the quantization was omitted.
    
    """
    if bin_width <= 0.:
        raise ValueError('The quantization bin width is not strictly positive.')
    
    # Let's assume that samples were quantized
    # using a uniform scalar quantization with
    # quantization bin width 0.5, giving rise to
    # the quantized samples. The quantized samples are
    # passed as the 1st argument of `count_symbols` and
    # 0.25 is passed as the 2nd argument. The assertion
    # below cannot detect that 0.25 is not the right
    # quantization bin width. The assertion below can
    # only ensure that the quantization was not omitted.
    numpy.testing.assert_almost_equal(bin_width*numpy.round(quantized_samples/bin_width),
                                      quantized_samples,
                                      decimal=10,
                                      err_msg='The quantization was omitted.')
    minimum = numpy.amin(quantized_samples)
    maximum = numpy.amax(quantized_samples)
    nb_edges = int(numpy.round((maximum - minimum)/bin_width)) + 2
    
    # The 3rd argument of the function `numpy.linspace`
    # must be an integer. It is optional.
    bin_edges = numpy.linspace(minimum - 0.5*bin_width,
                               maximum + 0.5*bin_width,
                               num=nb_edges)
    
    # In the function `numpy.histogram`, `quantized_samples`
    # is flattened to compute the histogram.
    return numpy.histogram(quantized_samples, bins=bin_edges)[0]

def crop_option_2d(luminance_uint8, width_crop, is_random):
    """Crops the luminance image randomly if it is the random option. Crops its center otherwise.
    
    Parameters
    ----------
    luminance_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Luminance image.
    width_crop : int
        Width of the crop.
    is_random : bool
        Is it the random option?
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Crop of the luminance image.
    
    Raises
    ------
    TypeError
        If `luminance_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `luminance_uint8.ndim` is not equal to 2.
    ValueError
        If either the height or the width of
        the luminance image is not larger than
        the width of the crop.
    
    """
    if luminance_uint8.dtype != numpy.uint8:
        raise TypeError('`luminance_uint8.dtype` is not equal to `numpy.uint8`.')
    if luminance_uint8.ndim != 2:
        raise ValueError('`luminance_uint8.ndim` is not equal to 2.')
    (height_image, width_image) = luminance_uint8.shape
    if height_image < width_crop or width_image < width_crop:
        raise ValueError('Either the height or the width of the luminance image is not larger than the width of the crop.')
    if is_random:
        i = numpy.random.choice(height_image - width_crop + 1)
        j = numpy.random.choice(width_image - width_crop + 1)
    else:
        i = (height_image - width_crop)//2
        j = (width_image - width_crop)//2
    return luminance_uint8[i:i + width_crop, j:j + width_crop]

def crop_repeat_2d(image_uint8, row_top_left, column_top_left):
    """Crops the image and repeats the pixels of the crop.
    
    A 80x80 crop is extracted from the image.
    Then, the pixels of the crop are repeated
    2 times horizontally and 2 times vertically.
    The resulting crop is 160x160.
    
    Parameters
    ----------
    image_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Image to be cropped.
    row_top_left : int
        Row of the image pixel at the
        top-left of the crop.
    column_top_left : int
        Column of the image pixel at the
        top-left of the crop.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.uint8`.
        160x160 crop.
    
    Raises
    ------
    TypeError
        If `image_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `image_uint8.ndim` is not equal to 2.
    ValueError
        If `image_uint8.shape[0]` is not strictly
        larger than `row_top_left + 80`.
    ValueError
        If `image_uint8.shape[1]` is not strictly
        larger than `column_top_left + 80`.
    
    """
    if image_uint8.dtype != numpy.uint8:
        raise TypeError('`image_uint8.dtype` is not equal to `numpy.uint8`.')
    if image_uint8.ndim != 2:
        raise ValueError('`image_uint8.ndim` is not equal to 2.')
    if row_top_left + 80 >= image_uint8.shape[0]:
        raise ValueError('`image_uint8.shape[0]` is not strictly larger than `row_top_left + 80`.')
    if column_top_left + 80 >= image_uint8.shape[1]:
        raise ValueError('`image_uint8.shape[1]` is not strictly larger than `column_top_left + 80`.')
    crop_uint8 = image_uint8[row_top_left:row_top_left + 80, column_top_left:column_top_left + 80]
    return numpy.repeat(numpy.repeat(crop_uint8, 2, axis=0), 2, axis=1)

def discrete_entropy(quantized_samples, bin_width):
    """Computes the entropy of the quantized samples.
    
    In exact terms, the quantized samples are viewed
    as realizations of a discrete random variable.
    The estimated entropy of the discrete random
    variable is computed using the quantized samples.
    The term "discrete" in the function name
    emphasizes that we are not talking about
    differential entropy.
    
    Parameters
    ----------
    quantized_samples : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
        Quantized samples.
    bin_width : float
        Quantization bin width. Previously, samples
        were quantized using a uniform scalar quantization
        with quantization bin width `bin_width`, giving
        rise to `quantized_samples`.
    
    Returns
    -------
    numpy.float64
        Entropy of the quantized samples.
    
    Raises
    ------
    ValueError
        If the entropy is not positive.
    ValueError
        If the entropy is not smaller than
        its upper bound.
    
    """
    hist = count_symbols(quantized_samples, bin_width)
    hist_non_zero = numpy.extract(hist != 0, hist)
    
    # `hist_non_zero.dtype` is equal to `numpy.int64`.
    # In Python 2.x, the standard division between
    # two integers returns an integer whereas, in
    # Python 3.x, the standard division between
    # two integers returns a float.
    frequency = hist_non_zero.astype(numpy.float64)/numpy.sum(hist_non_zero)
    disc_entropy = -numpy.sum(frequency*numpy.log2(frequency))
    if disc_entropy < 0.:
        raise ValueError('The entropy is not positive.')
    if disc_entropy > numpy.log2(hist_non_zero.size):
        raise ValueError('The entropy is not smaller than its upper bound.')
    return disc_entropy

def float_to_str(float_in):
    """Converts the float into a string.
    
    During the conversion, "." is replaced
    by "dot" if the float is not a whole
    number and "-" is replaced by "minus".
    
    Parameters
    ----------
    float_in : float
        Float to be converted.
    
    Returns
    -------
    str
        String resulting from
        the conversion.
    
    """
    if float_in.is_integer():
        str_in = str(int(float_in))
    else:
        str_in = str(float_in).replace('.', 'dot')
    return str_in.replace('-', 'minus')

def histogram(data, title, path):
    """Creates a histogram of the data and saves the histogram.
    
    Parameters
    ----------
    data : numpy.ndarray
        1D array.
        Data.
    title : str
        Title of the histogram.
    path : str
        Path to the saved histogram. The
        path must end with ".png".
    
    """
    plt.hist(data, bins=60)
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def jensen_shannon_divergence(probs_0, probs_1):
    """Computes the Jensen-Shannon divergence between the two probability distributions.
    
    Parameters
    ----------
    probs_0 : numpy.ndarray
        1D array with data-type `numpy.float64`.
        1st probability distribution.
    probs_1 : numpy.ndarray
        1D array with data-type `numpy.float64`.
        2nd probability distribution. `probs_0[i]`
        and `probs_1[i]` are two probabilities of
        the same symbol.
    
    Returns
    -------
    numpy.float64
        Jensen-Shannon divergence between the two
        probability distributions.
    
    Raises
    ------
    ValueError
        If a probability in `probs_0` does not belong to ]0.0, 1.0[.
    ValueError
        If a probability in `probs_1` does not belong to ]0.0, 1.0[.
    ValueError
        If the probabilities in `probs_0` do not sum to 1.0.
    ValueError
        If the probabilities in `probs_1` do not sum to 1.0.
    ValueError
        If the Jensen-Shannon divergence is not positive.
    ValueError
        If the Jensen-Shannon divergence is not
        smaller than 1.0.
    
    """
    if numpy.any(probs_0 <= 0.) or numpy.any(probs_0 >= 1.):
        raise ValueError('A probability in `probs_0` does not belong to ]0.0, 1.0[.')
    if numpy.any(probs_1 <= 0.) or numpy.any(probs_1 >= 1.):
        raise ValueError('A probability in `probs_1` does not belong to ]0.0, 1.0[.')
    if abs(numpy.sum(probs_0).item() - 1.) >= 1.e-9:
        raise ValueError('The probabilities in `probs_0` do not sum to 1.0.')
    if abs(numpy.sum(probs_1).item() - 1.) >= 1.e-9:
        raise ValueError('The probabilities in `probs_1` do not sum to 1.0.')
    denominator = 0.5*(probs_0 + probs_1)
    divergence = 0.5*numpy.sum(probs_0*numpy.log2(probs_0/denominator) + probs_1*numpy.log2(probs_1/denominator))
    if divergence < 0.:
        raise ValueError('The Jensen-Shannon divergence is not positive.')
    if divergence > 1.:
        raise ValueError('The Jensen-Shannon divergence is not smaller than 1.0.')
    return divergence

def normed_histogram(data, grid, pdfs, titles, paths):
    """Creates a normed histogram for each set of data and saves the normed histograms.
    
    Each set of data is associated to a
    sampled probability density function.
    For each set of data, a normed histogram
    is created and the associated sampled probability
    density function is drawn. Then, the normed
    histograms are saved.
    
    Parameters
    ----------
    data : numpy.ndarray
        4D array.
        Data. `data[:, :, :, i]` is the ith set of data.
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
    pdfs : numpy.ndarray
        2D array with data-type `numpy.float32` or `numpy.float64`.
        Sampled probability density functions. `pdfs[i, :]`
        is the sampled probability density function
        associated to `data[:, :, :, i]`.
    titles : list
        `titles[i]` is the title of the ith
        normed histogram.
    paths : list
        `paths[i]` is the path to the ith 
        saved normed histogram. Each path
        must end with ".png".
    
    Raises
    ------
    ValueError
        If `data.ndim` is not equal to 4.
    ValueError
        If `grid.ndim` is not equal to 1.
    ValueError
        If `pdfs.ndim` is not equal to 2.
    ValueError
        If `pdfs.shape[1]` is not equal
        to `grid.size`.
    ValueError
        If `pdfs.shape[0]` is not equal
        to `data.shape[3]`.
    ValueError
        If `len(titles)` is not equal to `data.shape[3]`.
    ValueError
        If `len(paths)` is not equal to `data.shape[3]`.
    
    """
    if data.ndim != 4:
        raise ValueError('`data.ndim` is not equal to 4.')
    if grid.ndim != 1:
        raise ValueError('`grid.ndim` is not equal to 1.')
    if pdfs.ndim != 2:
        raise ValueError('`pdfs.ndim` is not equal to 2.')
    if pdfs.shape[1] != grid.size:
        raise ValueError('`pdfs.shape[1]` is not equal to `grid.size`.')
    nb_maps = data.shape[3]
    if pdfs.shape[0] != nb_maps:
        raise ValueError('`pdfs.shape[0]` is not equal to `data.shape[3]`.')
    if len(titles) != nb_maps:
        raise ValueError('`len(titles)` is not equal to `data.shape[3]`.')
    if len(paths) != nb_maps:
        raise ValueError('`len(paths)` is not equal to `data.shape[3].`')
    data_2d = numpy.transpose(numpy.reshape(data, (-1, nb_maps)))
    for i in range(nb_maps):
        
        # `hist.dtype` is equal to `numpy.float64`
        # as the histogram is normalized.
        hist, bin_edges = numpy.histogram(data_2d[i, :],
                                          bins=60,
                                          density=True)
        plt.bar(bin_edges[0:60],
                hist,
                width=bin_edges[1] - bin_edges[0],
                align='edge',
                color='blue')
        plt.plot(grid,
                 pdfs[i, :],
                 color='red')
        plt.title(titles[i])
        plt.savefig(paths[i])
        plt.clf()

def plot_graphs(x_values, y_values, x_label, y_label, legend, colors, title, path):
    """Overlays several graphs in the same plot and saves the plot.
    
    Parameters
    ----------
    x_values : numpy.ndarray
        1D array.
        x-axis values.
    y_values : numpy.ndarray
        2D array.
        `y_values[i, :]` contains the
        y-axis values of the ith graph.
    x_label : str
        x-axis label.
    y_label : str
        y-axis label.
    legend : list
        `legend[i]` is a string describing
        the ith graph.
    colors : list
        `colors[i]` is a string characterizing
        the color of the ith graph.
    title : str
        Title of the plot.
    path : str
        Path to the saved plot. The path
        must end with ".png".
    
    Raises
    ------
    ValueError
        If `x_values.ndim` is not equal to 1.
    ValueError
        If `y_values.ndim` is not equal to 2.
    ValueError
        If `x_values.size` is not equal to
        `y_values.shape[1]`.
    ValueError
        If `len(legend)` is not equal to `y_values.shape[0]`.
    ValueError
        If `len(colors)` is not equal to `y_values.shape[0]`.
    
    """
    if x_values.ndim != 1:
        raise ValueError('`x_values.ndim` is not equal to 1.')
    if y_values.ndim != 2:
        raise ValueError('`y_values.ndim` is not equal to 2.')
    (nb_graphs, nb_y) = y_values.shape
    if x_values.size != nb_y:
        raise ValueError('`x_values.size` is not equal to `y_values.shape[1]`.')
    if nb_graphs != len(legend):
        raise ValueError('`len(legend)` is not equal to `y_values.shape[0]`.')
    if nb_graphs != len(colors):
        raise ValueError('`len(colors)` is not equal to `y_values.shape[0]`.')
    
    # Matplotlib is forced to display only
    # whole numbers on the x-axis if the
    # x-axis values are integers. Matplotlib
    # is also forced to display only whole
    # numbers on the y-axis if the y-axis
    # values are integers.
    current_axis = plt.figure().gca()
    if numpy.issubdtype(x_values.dtype, numpy.integer):
        current_axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if numpy.issubdtype(y_values.dtype, numpy.integer):
        current_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handle = []
    
    # The function `plt.plot` returns a list.
    for i in range(nb_graphs):
        handle.append(plt.plot(x_values, y_values[i, :], colors[i])[0])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handle, legend)
    plt.savefig(path)
    plt.clf()

def psnr_2d(reference_uint8, reconstruction_uint8):
    """Computes the PSNR between the luminance image and its reconstruction.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Luminance image.
    reconstruction_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Reconstruction of the luminance image.
    
    Returns
    -------
    numpy.float64
        PSNR between the luminance image
        and its reconstruction.
    
    Raises
    ------
    TypeError
        If `reference_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        If `reconstruction_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `reference_uint8.ndim` is not equal to 2.
    ValueError
        If `reference_uint8.shape` is not equal
        to `reconstruction_uint8.shape`.
    ValueError
        If the mean squared error between the luminance
        image and its reconstruction is 0.
    
    """
    if reference_uint8.dtype != numpy.uint8:
        raise TypeError('`reference_uint8.dtype` is not equal to `numpy.uint8`.')
    if reconstruction_uint8.dtype != numpy.uint8:
        raise TypeError('`reconstruction_uint8.dtype` is not equal to `numpy.uint8`.')
    if reference_uint8.ndim != 2:
        raise ValueError('`reference_uint8.ndim` is not equal to 2.')
    if reference_uint8.shape != reconstruction_uint8.shape:
        raise ValueError('`reference_uint8.shape` is not equal to `reconstruction_uint8.shape`.')
    reference_float64 = reference_uint8.astype(numpy.float64)
    reconstruction_float64 = reconstruction_uint8.astype(numpy.float64)
    mse = numpy.mean((reference_float64 - reconstruction_float64)**2)
    
    # A perfect reconstruction is impossible
    # in lossy compression.
    if mse == 0.:
        raise ValueError('The mean squared error between the luminance image and its reconstruction is 0.')
    return 10.*numpy.log10((255.**2)/mse)

def quantize_per_map(data, bin_widths):
    """Quantizes each map of data separately using a uniform scalar quantization.
    
    Parameters
    ----------
    data : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Data. `data[:, :, :, i]` is the ith map
        of data to be quantized.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths. `data[:, :, :, i]`
        is quantized using a uniform scalar quantization
        with quantization bin width `bin_widths[i]`.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.float32`.
        Quantized data.
    
    Raises
    ------
    ValueError
        If `data.ndim` is not equal to 4.
    ValueError
        If `bin_widths.ndim` is not equal to 1.
    ValueError
        If `bin_widths.size` is not equal
        to `data.shape[3]`.
    ValueError
        If a quantization bin width is not
        strictly positive.
    
    """
    if data.ndim != 4:
        raise ValueError('`data.ndim` is not equal to 4.')
    if bin_widths.ndim != 1:
        raise ValueError('`bin_widths.ndim` is not equal to 1.')
    (nb_examples, height_map, width_map, nb_maps) = data.shape
    if bin_widths.size != nb_maps:
        raise ValueError('`bin_widths.size` is not equal to `data.shape[3]`.')
    if numpy.any(bin_widths <= 0.):
        raise ValueError('A quantization bin width is not strictly positive.')
    tiled_bin_widths = numpy.tile(numpy.reshape(bin_widths, (1, 1, 1, nb_maps)),
                                  (nb_examples, height_map, width_map, 1))
    return tiled_bin_widths*numpy.round(data/tiled_bin_widths)

def rate_3d(quantized_latent_float32, bin_widths, h_in, w_in):
    """Computes the rate associated to the compression of a luminance image into the quantized latent variables.
    
    Previously, a luminance image was transformed
    into latent variables and the latent variables
    were quantized, giving rise to the quantized
    latent variables.
    
    Parameters
    ----------
    quantized_latent_float32 : numpy.ndarray
        3D array with data-type `numpy.float32`.
        Quantized latent variables.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Quantization bin widths. Previously, a uniform
        scalar quantization with quantization bin width
        `bin_widths[i]` was applied to latent variables,
        giving rise to `quantized_latent_float32[:, :, i]`.
    h_in : int
        Height of the luminance image.
    w_in : int
        Width of the luminance image.
    
    Returns
    -------
    numpy.float64
        Rate.
    
    Raises
    ------
    ValueError
        If `quantized_latent_float32.ndim` is not equal to 3.
    ValueError
        If `bin_widths.ndim` is not equal to 1.
    ValueError
        If `bin_widths.size` is not equal to
        `quantized_latent_float32.shape[2]`.
    
    """
    if quantized_latent_float32.ndim != 3:
        raise ValueError('`quantized_latent_float32.ndim` is not equal to 3.')
    if bin_widths.ndim != 1:
        raise ValueError('`bin_widths.ndim` is not equal to 1.')
    (height_map, width_map, nb_maps) = quantized_latent_float32.shape
    if bin_widths.size != nb_maps:
        raise ValueError('`bin_widths.size` is not equal to `quantized_latent_float32.shape[2]`.')
    cumulated_rate = 0.
    for i in range(nb_maps):
        
        # The function `discrete_entropy` checks
        # that the user did not omit the uniform
        # scalar quantization.
        # In the function `discrete_entropy`,
        # `quantized_latent_float32[:, :, i]` is flattened
        # to compute the entropy.
        disc_entropy = discrete_entropy(quantized_latent_float32[:, :, i],
                                        bin_widths[i].item())
        cumulated_rate += disc_entropy*height_map*width_map
    return cumulated_rate/(h_in*w_in)

def read_image_mode(path, mode):
    """Reads the image if its mode matches the given mode.

    Parameters
    ----------
    path : str
        Path to the image to be read.
    mode : str
        Given mode. The two most common modes
        are 'RGB' and 'L'.

    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.uint8`.
        Image.

    Raises
    ------
    ValueError
        If the image mode is not equal to `mode`.

    """
    image = PIL.Image.open(path)
    if image.mode != mode:
        raise ValueError('The image mode is {0} whereas the given mode is {1}.'.format(image.mode, mode))
    return numpy.asarray(image)

def rgb_to_ycbcr(rgb_uint8):
    """Converts the RGB image to YCbCr.
    
    Here, the conversion is ITU-R BT.601. This
    means that, if the pixels of the RGB image
    span the whole range [|0, 255|], the pixels
    of the luminance channel span the range [|16, 235|]
    and the pixels of the chrominance channels span
    the range [|16, 240|]. `rgb_to_ycbcr` is equivalent
    to the function `rgb2ycbcr` in Matlab. Note that
    the OpenCV function `cvtColor` with the code
    `CV_BGR2YCrCb` is different as it is the ITU-T T.871
    conversion, <http://www.itu.int/rec/T-REC-T.871>. The
    ITU-T T.871 conversion is used in JPEG.
    
    Parameters
    ----------
    rgb_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        RGB image.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        YCbCr image.
    
    Raises
    ------
    TypeError
        If `rgb_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `rgb_uint8.ndim` is not equal to 3.
    ValueError
        If `rgb_uint8.shape[2]` is not equal to 3.
    
    """
    if rgb_uint8.dtype != numpy.uint8:
        raise TypeError('`rgb_uint8.dtype` is not equal to `numpy.uint8`.')
    if rgb_uint8.ndim != 3:
        raise ValueError('`rgb_uint8.ndim` is not equal to 3.')
    if rgb_uint8.shape[2] != 3:
        raise ValueError('`rgb_uint8.shape[2]` is not equal to 3.')
    rgb_float64 = rgb_uint8.astype(numpy.float64)
    y_float64 = 16. \
                + (65.481/255.)*rgb_float64[:, :, 0] \
                + (128.553/255.)*rgb_float64[:, :, 1] \
                + (24.966/255.)*rgb_float64[:, :, 2]
    cb_float64 = 128. \
                 - (37.797/255.)*rgb_float64[:, :, 0] \
                 - (74.203/255.)*rgb_float64[:, :, 1] \
                 + (112./255.)*rgb_float64[:, :, 2]
    cr_float64 = 128. \
                 + (112./255.)*rgb_float64[:, :, 0] \
                 - (93.786/255.)*rgb_float64[:, :, 1] \
                 - (18.214/255.)*rgb_float64[:, :, 2]
    ycbcr_float64 = numpy.stack((y_float64, cb_float64, cr_float64), axis=2)
    
    # Before casting from `numpy.float64` to `numpy.uint8`,
    # all floats are clipped to [0., 255.] and rounded to
    # the nearest whole number.
    return numpy.round(ycbcr_float64.clip(min=0., max=255.)).astype(numpy.uint8)

def save_image(path, array_uint8):
    """Saves the array as an image.
    
    `scipy.misc.imsave` is deprecated in Scipy 1.0.0.
    `scipy.misc.imsave` will be removed in Scipy 1.2.0.
    `save_image` replaces `scipy.misc.imsave`.
    
    Parameters
    ----------
    path : str
        Path to the saved image.
    array_uint8 : numpy.ndarray
        Array with data-type `numpy.uint8`.
        Array to be saved as an image.

    Raises
    ------
    TypeError
        If `array_uint8.dtype` is not equal to `numpy.uint8`.

    """
    if array_uint8.dtype != numpy.uint8:
        raise TypeError('`array_uint8.dtype` is not equal to `numpy.uint8`.')
    image = PIL.Image.fromarray(array_uint8)
    image.save(path)

def subdivide_set(nb_examples, batch_size):
    """Computes the number of mini-batches in the set of examples.
    
    Parameters
    ----------
    nb_examples : int
        Number of examples.
    batch_size : int
        Size of the mini-batches.
    
    Returns
    -------
    int
        Number of mini-batches in
        the set of examples.
    
    Raises
    ------
    ValueError
        If `nb_examples` is not divisible by `batch_size`.
    
    """
    if nb_examples % batch_size != 0:
        raise ValueError('`nb_examples` is not divisible by `batch_size`.')
    return nb_examples//batch_size

def tile_cauchy(grid, reps):
    """Computes the Cauchy probability density function at the sampling points and repeats the result.
    
    Parameters
    ----------
    grid : numpy.ndarray
        1D array with data-type `numpy.float32`.
        Grid storing the sampling points.
    reps : int
        Number of times the sampled Cauchy
        probability density function is repeated.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float32`.
        Each row contains the same sampled
        Cauchy probability density function.
    
    """
    pdf_float32 = scipy.stats.distributions.cauchy.pdf(grid).astype(numpy.float32)
    return numpy.tile(numpy.expand_dims(pdf_float32, axis=0), (reps, 1))

def untar_archive(path_to_root, path_to_tar):
    """Extracts the archive to the given folder.
    
    Parameters
    ----------
    path_to_root : str
        Path to the given folder.
    path_to_tar : str
        Path to the archive to be extracted.
        The path must end with ".tar".
    
    """
    with tarfile.open(path_to_tar, 'r') as file:
        file.extractall(path=path_to_root)

def visualize_crops(image_uint8, positions_top_left, paths):
    """Crops the image several times, repeats the pixels of each crop and saves the resulting crops.
    
    Several 80x80 crops are extracted from the image.
    Then, the pixels of each crop are repeated 2 times
    horizontally and 2 times vertically. Finally, the
    resulting 160x160 crops are saved.
    
    Parameters
    ----------
    image_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Image to be cropped.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        `positions_top_left[:, i]` contains
        the row and the column of the image
        pixel at the top-left of the ith crop.
    paths : list
        `paths[i]` is the path to the ith saved
        crop. Each path must end with ".png".
    
    Raises
    ------
    ValueError
        If `positions_top_left.ndim` is not equal to 2.
    ValueError
        If `positions_top_left.shape[0]` is not equal to 2.
    ValueError
        If `len(paths)` is not equal to `positions_top_left.shape[1]`.
    
    """
    if positions_top_left.ndim != 2:
        raise ValueError('`positions_top_left.ndim` is not equal to 2.')
    if positions_top_left.shape[0] != 2:
        raise ValueError('`positions_top_left.shape[0]` is not equal to 2.')
    nb_crops = positions_top_left.shape[1]
    if len(paths) != nb_crops:
        raise ValueError('`len(paths)` is not equal to `positions_top_left.shape[1]`.')
    for i in range(nb_crops):
        
        # The function `crop_repeat_2d` checks that
        # `image_uint8.dtype` is equal to `numpy.uint8`
        # and `image_uint8.ndim` is equal to 2.
        crop_uint8 = crop_repeat_2d(image_uint8,
                                    positions_top_left[0, i].item(),
                                    positions_top_left[1, i].item())
        save_image(paths[i],
                   crop_uint8)

def visualize_luminances(luminances_uint8, nb_vertically, path):
    """Arranges the luminance images in a single image and saves the single image.
    
    Parameters
    ----------
    luminances_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        Luminance images. `luminances_uint8[i, :, :, :]`
        is the ith luminance image. The 4th dimension
        of `luminances_uint8` must be equal to 1.
    nb_vertically : int
        Number of luminance images per column
        in the single image.
    path : str
        Path to the saved single image. The path
        must end with ".png".
    
    Raises
    ------
    TypeError
        If `luminances_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `luminances_uint8.ndim` is not equal to 4.
    ValueError
        If `luminances_uint8.shape[3]` is not equal to 1.
    ValueError
        If `luminances_uint8.shape[0]` is
        not divisible by `nb_vertically`.
    
    """
    if luminances_uint8.dtype != numpy.uint8:
        raise TypeError('`luminances_uint8.dtype` is not equal to `numpy.uint8`.')
    if luminances_uint8.ndim != 4:
        raise ValueError('`luminances_uint8.ndim` is not equal to 4.')
    (nb_images, height_image, width_image, nb_channels) = luminances_uint8.shape
    if nb_channels != 1:
        raise ValueError('`luminances_uint8.shape[3]` is not equal to 1.')
    if nb_images % nb_vertically != 0:
        raise ValueError('`luminances_uint8.shape[0]` is not divisible by `nb_vertically`.')
    
    # `nb_horizontally` has to be an integer.
    nb_horizontally = nb_images//nb_vertically
    image_uint8 = 255*numpy.ones((nb_vertically*(height_image + 1) + 1,
        nb_horizontally*(width_image + 1) + 1), dtype=numpy.uint8)
    for i in range(nb_vertically):
        for j in range(nb_horizontally):
            image_uint8[i*(height_image + 1) + 1:(i + 1)*(height_image + 1),
                j*(width_image + 1) + 1:(j + 1)*(width_image + 1)] = \
                luminances_uint8[i*nb_horizontally + j, :, :, 0]
    save_image(path,
               image_uint8)

def visualize_representation(representation_float32, nb_vertically, path):
    """Arranges the representation feature maps in a single image and saves the single image.
    
    Parameters
    ----------
    representation_float32 : numpy.ndarray
        3D array with data-type `numpy.float32`.
        Representation feature maps. `representation_float32[:, :, i]`
        is the ith representation feature map.
    nb_vertically : int
        Number of representation feature maps
        per column in the single image.
    path : str
        Path to the saved single image. The path
        must end with ".png".
    
    """
    min_r = numpy.amin(representation_float32)
    max_r = numpy.amax(representation_float32)
    representation_uint8 = numpy.round(255.*(representation_float32 - min_r)/(max_r - min_r)).astype(numpy.uint8)
    representation_swapaxes_uint8 = numpy.swapaxes(numpy.swapaxes(representation_uint8, 1, 2), 0, 1)
    visualize_luminances(numpy.expand_dims(representation_swapaxes_uint8, axis=3),
                         nb_vertically,
                         path)

def visualize_rotated_luminance(luminance_before_rotation_uint8, is_rotated, positions_top_left, paths):
    """Rotates the luminance image if required and crops the result several times.
    
    The possibly rotated luminance image and the
    crops are saved.
    
    Parameters
    ----------
    luminance_before_rotation_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Luminance image before the rotation.
    is_rotated : bool
        Is the luminance image rotated?
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        `positions_top_left[:, i]` contains
        the row and the column of the image
        pixel at the top-left of the ith crop.
    paths : list
        The 1st string in this list is the path
        to the saved rotated luminance image. Each
        string in this list from the 2nd to the last
        is the path to a saved crop. Each path must
        end with ".png".
    
    """
    if is_rotated:
        image_uint8 = numpy.rot90(luminance_before_rotation_uint8, k=3).copy()
    else:
        image_uint8 = luminance_before_rotation_uint8.copy()
    
    # The function `visualize_crops` checks that
    # `image_uint8.dtype` is equal to `numpy.uint8`
    # and `image_uint8.ndim` is equal to 2.
    visualize_crops(image_uint8,
                    positions_top_left,
                    paths[1:])
    save_image(paths[0],
               image_uint8)

def visualize_weights(weights, nb_vertically, path):
    """Arranges the weight filters in a single image and saves the single image.
    
    Parameters
    ----------
    weights : numpy.ndarray
        4D array with data-type `numpy.float32`.
        Weight filters. `weights[:, :, :, i]` is
        the ith weight filter. The 3rd dimension
        of `weights` must be equal to 1.
    nb_vertically : int
        Number of weight filters per column
        in the single image.
    path : str
        Path to the saved single image. The path
        must end with ".png".
    
    """
    min_w = numpy.amin(weights)
    max_w = numpy.amax(weights)
    weights_uint8 = numpy.round(255.*(weights - min_w)/(max_w - min_w)).astype(numpy.uint8)
    weights_swapaxes_uint8 = numpy.swapaxes(numpy.swapaxes(numpy.swapaxes(weights_uint8, 2, 3), 1, 2), 0, 1)
    visualize_luminances(weights_swapaxes_uint8,
                         nb_vertically,
                         path)


