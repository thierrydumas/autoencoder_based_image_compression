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

# The functions are sorted in
# alphabetic order.

def approximate_entropy(y_tilde, parameters, nb_points_per_interval, nb_intervals_per_side, bin_width):
    """Computes the approximate entropy of the quantized latent variables.
    
    The differential entropy of the latent variables
    perturbed by uniform noise is first computed using
    the piecewise linear function. Then, the differential
    entropy of the latent variables perturbed by uniform
    noise is converted into the approximate entropy of the
    quantized latent variables using the quantization bin
    width.
    
    Parameters
    ----------
    y_tilde : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Latent variables perturbed by uniform noise.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the probability density function of the latent
        variables perturbed by uniform noise.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    bin_width : float
        Quantization bin width.
    
    Returns
    -------
    numpy.float64
        Approximate entropy of the quantized latent
        variables.
    
    Raises
    ------
    ValueError
        If the quantization bin width is not
        strictly positive.
    ValueError
        If the approximate entropy of the quantized
        latent variables is not positive.
    
    """
    if bin_width <= 0.:
        raise ValueError('The quantization bin width is not strictly positive.')
    diff_entropy = differential_entropy(y_tilde,
                                        parameters,
                                        nb_points_per_interval,
                                        nb_intervals_per_side)
    approx_entropy = diff_entropy - numpy.log2(bin_width)
    if approx_entropy < 0.:
        raise ValueError('The approximate entropy of the quantized latent variables is not positive.')
    return approx_entropy

def approximate_probability(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the approximate probability of each sample.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from an unknown probability density
        function.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        Approximate probabilities. Its ith element is
        the approximate probability of `samples[i]`.
    
    """
    idx_linear_piece = index_linear_piece(samples,
                                          nb_points_per_interval,
                                          nb_intervals_per_side)
    right = numpy.take(parameters, idx_linear_piece + 1)
    left = numpy.take(parameters, idx_linear_piece)
    left_bound = numpy.floor(nb_points_per_interval*samples)/nb_points_per_interval
    return (right - left)*(samples - left_bound)*nb_points_per_interval + left

def area_under_piecewise_linear_function(parameters, nb_points_per_interval):
    """Computes the area under the piecewise linear function.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    
    Returns
    -------
    numpy.float64
        Area under the piecewise linear function.
    
    """
    nb_points = parameters.size
    return 0.5*(parameters[0] + parameters[nb_points - 1] +
        2.*numpy.sum(parameters[1:nb_points - 1]))/nb_points_per_interval

def cast_float_to_uint8(array_float):
    """Casts the array elements from float to 8-bit unsigned integer.
    
    The array elements are clipped to [0., 255.],
    rounded to the nearest whole number and cast
    from float to 8-bit unsigned integer.
    
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
    return numpy.round(array_float.clip(min=0., max=255.)).astype(numpy.uint8)

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
        from a probability density function were
        quantized using a uniform scalar quantization
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
    AssertionError
        If the quantization was omitted.
    
    """
    # Let's assume that samples from a probability
    # density function were quantized using a uniform
    # scalar quantization with quantization bin width
    # 0.5, giving rise to the quantized samples. The
    # quantized samples are passed as the 1st argument
    # of `count_symbols` and 0.25 is passed as the 2nd
    # argument. The assertion below cannot detect that
    # 0.25 is not the right quantization bin width. The
    # assertion below can only ensure that the quantization
    # was not omitted.
    numpy.testing.assert_almost_equal(quantization(quantized_samples, bin_width),
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

def count_zero_columns(array_2d):
    """Counts the number of zero columns in the 2D array.
    
    Parameters
    ----------
    array_2d : numpy.ndarray
        2D array.
    
    Returns
    -------
    int
        Number of zero columns in `array_2d`.
    
    Raises
    ------
    ValueError
        If `array_2d.ndim` is not equal to 2.
    
    """
    if array_2d.ndim != 2:
        raise ValueError('`array_2d.ndim` is not equal to 2.')
    nb_columns = array_2d.shape[1]
    return nb_columns - numpy.count_nonzero(numpy.sum(numpy.absolute(array_2d), axis=0))

def differential_entropy(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the differential entropy of the samples.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from an unknown probability density
        function.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.float64
        Differential entropy of the samples.
    
    """
    approximate_prob = approximate_probability(samples,
                                               parameters,
                                               nb_points_per_interval,
                                               nb_intervals_per_side)
    return -numpy.mean(numpy.log2(approximate_prob))

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
        from a probability density function were
        quantized using a uniform scalar quantization
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

def expand_parameters(parameters, low_projection, nb_points_per_interval, nb_added_per_side):
    """Expands the parameters of the piecewise linear function.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function
        before the expansion.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear function. Thanks to
        `low_projection`, the parameters of the piecewise
        linear function cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_added_per_side : int
        Number of unit intervals added to each side
        of the grid.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function
        after the expansion.
    
    Raises
    ------
    ValueError
        If `low_projection` is not strictly larger than 1.e-7.
    ValueError
        If `nb_added_per_side` is not strictly positive.
    
    """
    if low_projection <= 1.e-7:
        raise ValueError('`low_projection` is not strictly larger than 1.e-7.')
    if nb_added_per_side <= 0:
        raise ValueError('`nb_added_per_side` is not striclty positive.')
    piece = low_projection*numpy.ones(nb_points_per_interval*nb_added_per_side)
    return numpy.concatenate((piece, parameters, piece), axis=0)

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

def gradient_density_approximation(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the derivative (with respect to each parameter of the piecewise linear function) of the
    loss of the approximation of an unknown probability density function with the piecewise linear function.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from the unknown probability density
        function.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        Its ith element is the derivative (with respect to the
        ith parameter of the piecewise linear function) of the
        loss of the approximation of the unknown probability
        density function with the piecewise linear function.
    
    """
    nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
    nb_samples = samples.size
    
    # The elements of `idx_linear_piece` belong to
    # [|0, `nb_points` - 2|] as the index of the last
    # sampling point in the grid cannot appear in
    # `idx_linear_piece`.
    idx_linear_piece = index_linear_piece(samples,
                                          nb_points_per_interval,
                                          nb_intervals_per_side)
    gradients = numpy.zeros(nb_points)
    for i in range(nb_points):
        accumulation = 0.
        
        # `i` is equal to 0 for the 1st sampling
        # point in the grid.
        if i > 0:
            sub_left = numpy.extract(idx_linear_piece == i - 1, samples)
            
            # If `sub_left` is empty, nothing is
            # added to `accumulation`.
            if sub_left.size:
                
                # `left_bound` is the sampling point
                # of index `i` - 1 in the grid. Any element
                # of `sub_left` can be used to compute
                # `left_bound` via the formula below.
                left_bound = numpy.floor(sub_left[0]*nb_points_per_interval)/nb_points_per_interval
                accumulation += numpy.sum(sub_left - left_bound)
        
        # `i` is equal to `nb_points` - 1 for the
        # last sampling point in the grid.
        if i < nb_points - 1:
            sub_right = numpy.extract(idx_linear_piece == i, samples)
            
            # If `sub_right` is empty, nothing is
            # added to `accumulation`.
            if sub_right.size:
                
                # `right_bound` is the sampling point
                # of index `i` + 1 in the grid. Any element
                # of `sub_right` can be used to compute
                # `right_bound` via the formula below.
                right_bound = numpy.ceil(sub_right[0]*nb_points_per_interval)/nb_points_per_interval
                accumulation += numpy.sum(right_bound - sub_right)
        gradients[i] = -2.*nb_points_per_interval*accumulation/nb_samples
    return gradients + 2.*parameters/nb_points_per_interval

def gradient_entropy(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the derivative of the differential entropy of the samples with respect to each sample.
    
    Parameters
    ----------
    samples : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Samples from an unknown probability density
        function.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float64`.
        The element at the position [i, j] in this
        array is the derivative of the differential
        entropy of the samples with respect to
        `samples[i, j]`.
    
    """
    # In the functions `approximate_entropy`,
    # `approximate_probability`, `differential_entropy`,
    # `gradient_density_approximation`, `index_linear_piece`
    # and `loss_density_approximation`, the 1st argument
    # is a 1D array. In the function `gradient_entropy`,
    # the 1st argument is a 2D array.
    (height_samples, width_samples) = samples.shape
    flattened_samples = samples.flatten()
    idx_linear_piece = index_linear_piece(flattened_samples,
                                          nb_points_per_interval,
                                          nb_intervals_per_side)
    right = numpy.take(parameters, idx_linear_piece + 1)
    left = numpy.take(parameters, idx_linear_piece)
    difference = right - left
    left_bound = numpy.floor(nb_points_per_interval*flattened_samples)/nb_points_per_interval
    flattened_gradients = -(1./(numpy.log(2.)*width_samples))*difference/(
        difference*(flattened_samples - left_bound) + (left/nb_points_per_interval))
    return numpy.reshape(flattened_gradients, (height_samples, width_samples))

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

def images_to_rows(images_uint8):
    """Reshapes each RGB image to a row.
    
    `images_to_rows` reverses the function `rows_to_images`.
    
    Parameters
    ----------
    images_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        RGB images. `images_uint8[:, :, :, i]`
        is the ith RGB image.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Its ith row contains the ith RGB image.
    
    Raises
    ------
    TypeError
        If `images_uint8.dtype` is not equal to
        `numpy.uint8`.
    ValueError
        If `images_uint8.shape[2]` is not equal to 3.
    
    """
    if images_uint8.dtype != numpy.uint8:
        raise TypeError('`images_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If `images_uint8.ndim` is not equal to 4,
    # the unpacking below raises a `ValueError`
    # exception.
    (height_image, width_image, nb_channels, nb_images) = images_uint8.shape
    if nb_channels != 3:
        raise ValueError('`images_uint8.shape[2]` is not equal to 3.')
    rows_uint8 = numpy.zeros((nb_images, 3*height_image*width_image), dtype=numpy.uint8)
    for i in range(nb_images):
        tuple_rgb = (
            numpy.reshape(images_uint8[:, :, 0, i], (1, height_image*width_image)),
            numpy.reshape(images_uint8[:, :, 1, i], (1, height_image*width_image)),
            numpy.reshape(images_uint8[:, :, 2, i], (1, height_image*width_image))
        )
        rows_uint8[i, :] = numpy.concatenate(tuple_rgb, axis=1)
    return rows_uint8

def index_linear_piece(samples, nb_points_per_interval, nb_intervals_per_side):
    """Finds the linear piece index of each sample.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from an unknown probability density
        function. The unknown probability density
        function is approximated by the piecewise
        linear function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.int64`.
        Linear piece indices. Its ith element is the
        linear piece index of `samples[i]`.
    
    Raises
    ------
    ValueError
        If a linear piece index is not positive.
    ValueError
        If a linear piece index exceeds the maximum
        possible linear piece index.
    
    """
    # The function `numpy.floor` returns an array
    # with data-type `numpy.float64`.
    idx_linear_piece = numpy.floor(nb_points_per_interval*samples).astype(numpy.int64) + \
        nb_points_per_interval*nb_intervals_per_side
    if numpy.any(idx_linear_piece < 0):
        raise ValueError('A linear piece index is not positive.')
    if numpy.any(idx_linear_piece > 2*nb_points_per_interval*nb_intervals_per_side - 1):
        raise ValueError('A linear piece index exceeds the maximum possible linear piece index.')
    return idx_linear_piece

def kl_divergence(z_mean, z_log_std_squared):
    """Computes the Kullback-Lieber divergence of the approximate posterior from the prior.
    
    Parameters
    ----------
    z_mean : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Mean of each latent variable normal distribution.
    z_log_std_squared : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Log standard deviation squared of each latent
        variable normal distribution.
    
    Returns
    -------
    numpy.float64
        Kullback-Lieber divergence of the approximate
        posterior from the prior.
    
    """
    return 0.5*numpy.mean(numpy.sum(-1. - z_log_std_squared + z_mean**2 +
        numpy.exp(z_log_std_squared), axis=1))

def leaky_relu(input):
    """Computes Leaky ReLU with slope 0.1.
    
    Parameters
    ----------
    input : numpy.ndarray
        Array with data-type `numpy.float64`.
        Input to Leaky ReLU.
        
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.float64`.
        Output from Leaky ReLU.
    
    """
    coefficients = numpy.ones(input.shape)
    coefficients[input < 0.] = 0.1
    return coefficients*input

def leaky_relu_derivative(input):
    """Computes the derivative of Leaky ReLU with respect to its input.
    
    The slope of Leaky ReLU is 0.1.
    
    Parameters
    ----------
    input : numpy.ndarray
        Array with data-type `numpy.float64`.
        Input to Leaky ReLU.
        
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.float64`.
        The element at the position [i, j, ...]
        in this array is the derivative of Leaky
        ReLU with respect to `input[i, j, ...]`.
    
    """
    coefficients = numpy.ones(input.shape)
    coefficients[input < 0.] = 0.1
    return coefficients

def loss_density_approximation(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the loss of the approximation of an unknown probability density function with a piecewise linear function.
    
    The loss is derived from the mean integrated squared
    error between the unknown probability density function
    and the piecewise linear function.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from the unknown probability density
        function.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    numpy.float64
        Loss of the approximation of the unknown probability
        density function with the piecewise linear function.
    
    """
    approximate_prob = approximate_probability(samples,
                                               parameters,
                                               nb_points_per_interval,
                                               nb_intervals_per_side)
    return -2.*numpy.mean(approximate_prob) + numpy.sum(parameters**2)/nb_points_per_interval

def loss_entropy_reconstruction(visible_units, y_tilde, reconstruction, parameters, nb_points_per_interval,
                                nb_intervals_per_side, bin_width, gamma):
    """Computes the entropy-reconstruction loss.
    
    The entropy-reconstruction loss contains two
    terms. The 1st term is the error between the
    visible units and their reconstruction. The
    2nd term is the scaled approximate entropy
    of the quantized latent variables.
    
    Parameters
    ----------
    visible_units : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Visible units.
    y_tilde : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Latent variables perturbed by uniform noise.
    reconstruction : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Reconstruction of the visible units.
    parameters : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function.
        The piecewise linear function approximates
        the probability density function of the latent
        variables perturbed by uniform noise.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    bin_width : float
        Quantization bin width.
    gamma : float
        Scaling coefficient.
    
    Returns
    -------
    numpy.float64
        Entropy-reconstruction loss.
    
    """
    approx_entropy = approximate_entropy(y_tilde.flatten(),
                                         parameters,
                                         nb_points_per_interval,
                                         nb_intervals_per_side,
                                         bin_width)
    rec_error = reconstruction_error(visible_units,
                                     reconstruction,
                                     True)
    return gamma*approx_entropy + rec_error

def mean_psnr(reference_uint8, reconstruction_uint8):
    """Computes the mean PSNR between the reference images and their reconstruction.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Reference images. `reference_uint8[i, :]`
        contains the ith reference image.
    reconstruction_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Reconstruction of the reference images.
        `reconstruction_uint8[i, :]` contains the
        reconstruction of the ith reference image.
    
    Returns
    -------
    numpy.float64
        Mean PSRN between the reference images
        and their reconstruction.
    
    Raises
    ------
    TypeError
        If `reference_uint8.dtype` is not equal to `numpy.uint8`.
    TypeError
        `reconstruction_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `reference_uint8.ndim` is not equal to 2.
    ValueError
        If `reference_uint8.shape` is not equal
        to `reconstruction_uint8.shape`.
    ValueError
        If the mean square error between a reference
        image and its reconstruction is equal to 0.
    
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
    mse = numpy.mean((reference_float64 - reconstruction_float64)**2, axis=1)
    
    # A perfect reconstruction is impossible
    # in lossy compression.
    if numpy.any(mse == 0.):
        raise ValueError('The mean square error between a reference image and its reconstruction is equal to 0.')
    return numpy.mean(10.*numpy.log10((255.**2)/mse))

def noise(nb_rows, nb_columns):
    """Draws noise from the uniform distribution of support [-0.5, 0.5].
    
    Parameters
    ----------
    nb_rows : int
        Number of rows of the output array.
    nb_columns : int
        Number of columns of the output array.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float64`.
        Noise from the uniform distribution
        of support [-0.5, 0.5].
    
    """
    return numpy.random.uniform(low=-0.5, high=0.5, size=(nb_rows, nb_columns))

def normed_histogram(data, grid, pdf, title, path):
    """Creates a normed histogram of the data and saves the normed histogram.
    
    A normed histogram of the data is created and
    the sampled probability density function
    associated to the data is drawn. Then, the
    normed histogram is saved.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data.
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
    pdf : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Sampled probability density function.
    title : str
        Title of the normed histogram.
    path : str
        Path to the saved normed histogram. The
        path must end with ".png".
    
    Raises
    ------
    ValueError
        If `grid.ndim` is not equal to 1.
    ValueError
        If `pdf.ndim` is not equal to 1.
    ValueError
        If `grid.size` is not equal to `pdf.size`.
    
    """
    if grid.ndim != 1:
        raise ValueError('`grid.ndim` is not equal to 1.')
    if pdf.ndim != 1:
        raise ValueError('`pdf.ndim` is not equal to 1.')
    if grid.size != pdf.size:
        raise ValueError('`grid.size` is not equal to `pdf.size`.')
    
    # In the function `numpy.histogram`, `data`
    # is flattened to compute the histogram.
    hist, bin_edges = numpy.histogram(data,
                                      bins=60,
                                      density=True)
    plt.bar(bin_edges[0:60],
            hist,
            width=bin_edges[1] - bin_edges[0],
            align='edge',
            color='blue')
    plt.plot(grid,
             pdf,
             color='red')
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def opposite_vlb(visible_units, z_mean, z_log_std_squared, reconstruction, alpha, is_continuous):
    """Computes the opposite of Kingma's approximation of the variational lower bound.
    
    To find out more, see the paper
    "Auto-encoding variational Bayes", written by
    Diederik P. Kingma and Max Welling (ICLR 2014).
    
    Parameters
    ----------
    visible_units : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Visible units.
    z_mean : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Mean of each latent variable normal distribution.
    z_log_std_squared : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Log standard deviation squared of each latent
        variable normal distribution.
    reconstruction : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Reconstruction of the visible units.
    alpha : float
        Scaling coefficient.
    is_continuous : bool
        Is each visible unit modeled as a continuous
        random variable with Gaussian probability
        density function?
    
    Returns
    -------
    numpy.float64
        Opposite of Kingma's approximation of the
        variational lower bound.
    
    """
    return alpha*kl_divergence(z_mean, z_log_std_squared) + \
        reconstruction_error(visible_units, reconstruction, is_continuous)

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
        If `x_values.size` is not equal to
        `y_values.shape[1]`.
    ValueError
        If `len(legend)` is not equal to `y_values.shape[0]`.
    ValueError
        If `len(colors)` is not equal to `y_values.shape[0]`.
    
    """
    if x_values.ndim != 1:
        raise ValueError('`x_values.ndim` is not equal to 1.')
    
    # If `y_values.ndim` is not equal to 2,
    # the unpacking below raises a `ValueError`
    # exception.
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

def quantization(samples, bin_width):
    """Quantizes the samples using a uniform scalar quantization.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Array whose data-type is smaller than
        `numpy.float` in type hierarchy.
        Samples to be quantized.
    bin_width : float
        Quantization bin width.
    
    Returns
    -------
    numpy.ndarray
        Array with the same shape and the same
        data-type as `samples`.
        Quantized samples.
    
    Raises
    ------
    TypeError
        If `samples.dtype` is not smaller than
        `numpy.float` in type hierarchy.
    ValueError
        If the quantization bin width is
        not strictly positive.
    
    """
    if not numpy.issubdtype(samples.dtype, numpy.float):
        raise TypeError('`samples.dtype` is not smaller than `numpy.float` in type hierarchy.')
    if bin_width <= 0.:
        raise ValueError('The quantization bin width is not strictly positive.')
    return bin_width*numpy.round(samples/bin_width)

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

def reconstruction_error(visible_units, reconstruction, is_continuous):
    """Computes the error between the visible units and their reconstruction.
    
    Parameters
    ----------
    visible_units : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Visible units.
    reconstruction : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Reconstruction of the visible units.
    is_continuous : bool
        Is each visible unit modeled as a
        continuous random variable with Gaussian
        probability density function?
    
    Returns
    -------
    numpy.float64
        Error between the visible units and
        their reconstruction.
    
    Raises
    ------
    ValueError
        If `is_continuous` is False and at least a
        coefficient of `visible_units` does not belong
        to [0, 1].
    ValueError
        If `is_continuous` is False and at least a
        coefficient `reconstruction` does not belong
        to ]0, 1[.
    
    """
    if is_continuous:
        
        # Compared to Kingma's approximation, a constant
        # term is removed and the standard deviation of
        # the Gaussian probability density function of
        # each visible unit is assumed to be 1.
        return 0.5*numpy.mean(numpy.sum((visible_units - reconstruction)**2, axis=1))
    else:
        if numpy.any(numpy.logical_or(visible_units < 0., visible_units > 1.)):
            raise ValueError('At least a coefficient of `visible_units` does not belong to [0, 1].')
        if numpy.any(numpy.logical_or(reconstruction <= 0., reconstruction >= 1.)):
            raise ValueError('At least a coefficient of `reconstruction` does not belong to ]0, 1[.')
        return -numpy.mean(numpy.sum(visible_units*numpy.log(reconstruction) + 
            (1. - visible_units)*numpy.log(1. - reconstruction), axis=1))

def relu(input):
    """Computes ReLU.
    
    Parameters
    ----------
    input : numpy.ndarray
        Array with data-type `numpy.float64`.
        Input to ReLU.
        
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.float64`.
        Output from ReLU.
    
    """
    return numpy.maximum(input, 0.)

def relu_derivative(input):
    """Computes the derivative of ReLU with respect to its input.
    
    Parameters
    ----------
    input : numpy.ndarray
        Array with data-type `numpy.float64`.
        Input to ReLU.
        
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.float64`.
        The element at the position [i, j, ...]
        in this array is the derivative of ReLU
        with respect to `input[i, j, ...]`.
    
    """
    return (input > 0.).astype(numpy.float64)

def rows_to_images(rows_uint8, height_image, width_image):
    """Reshapes each row to a RGB image.
    
    `rows_to_images` reverses the function `images_to_rows`.
    
    Parameters
    ----------
    rows_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        `rows_uint8[i, :]` contains the ith RGB image.
    height_image : int
        RGB image height.
    width_image : int
        RGB image width.
    
    Returns
    -------
    numpy.ndarray
        4D array with data-type `numpy.uint8`.
        RGB images.
    
    Raises
    ------
    TypeError
        If `rows_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `rows_uint8.shape[1]` is not equal
        to `3*height_image*width_image`.
    
    """
    if rows_uint8.dtype != numpy.uint8:
        raise TypeError('`rows_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # If `rows_uint8.ndim` is not equal to 2,
    # the unpacking below raises a `ValueError`
    # exception.
    (nb_images, nb_pixels_per_row) = rows_uint8.shape
    if nb_pixels_per_row != 3*height_image*width_image:
        raise ValueError('`rows_uint8.shape[1]` is not equal to `3*height_image*width_image`.')
    images_uint8 = numpy.zeros((height_image, width_image, 3, nb_images), dtype=numpy.uint8)
    for i in range(nb_images):
        images_uint8[:, :, 0, i] = \
            numpy.reshape(rows_uint8[i, 0:height_image*width_image],
                          (height_image, width_image))
        images_uint8[:, :, 1, i] = \
            numpy.reshape(rows_uint8[i, height_image*width_image:2*height_image*width_image],
                          (height_image, width_image))
        images_uint8[:, :, 2, i] = \
            numpy.reshape(rows_uint8[i, 2*height_image*width_image:3*height_image*width_image],
                          (height_image, width_image))
    return images_uint8

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

def sigmoid(input):
    """Computes the sigmoid function using a trick that avoids numerical overflow.
    
    Parameters
    ----------
    input : numpy.ndarray
        Array with data-type `numpy.float64`.
        Input to the sigmoid.
        
    Returns
    -------
    numpy.ndarray
        Array with data-type `numpy.float64`.
        Output from the sigmoid.
    
    """
    condition = input > 0.
    inverted_condition = numpy.invert(condition)
    positive_input = input[condition]
    negative_input = input[inverted_condition]
    sigmoid_activation = numpy.empty(input.shape)
    sigmoid_activation[condition] = 1./(1. + numpy.exp(-positive_input))
    sigmoid_activation[inverted_condition] = numpy.exp(negative_input)/(1. + numpy.exp(negative_input))
    return sigmoid_activation

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
        If `nb_examples` is not divisible
        by `batch_size`.
    
    """
    if nb_examples % batch_size != 0:
        raise ValueError('`nb_examples` is not divisible by `batch_size`.')
    return nb_examples//batch_size

def visualize_dead(quantized_samples, path):
    """Creates a heat map of the quantized samples using three colors and saves the map.
    
    The three colors are blue (strictly negative),
    black (zero) and red (strictly positive).
    
    Parameters
    ----------
    quantized_samples : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Quantized samples.
    path : str
        Path to the saved map. The path
        must end with ".png".
    
    """
    # If `quantized_samples.ndim` is not equal to 2,
    # the unpacking below raises a `ValueError` exception.
    (height_quantized_samples, width_quantized_samples) = quantized_samples.shape
    black_uint8 = numpy.zeros((height_quantized_samples, width_quantized_samples, 1), dtype=numpy.uint8)
    blue_uint8 = black_uint8.copy()
    blue_uint8[quantized_samples < 0.] = 255
    red_uint8 = black_uint8.copy()
    red_uint8[quantized_samples > 0.] = 255
    image_uint8 = numpy.concatenate((red_uint8, black_uint8, blue_uint8), axis=2)
    save_image(path,
               image_uint8)

def visualize_images(images_uint8, nb_vertically, path):
    """Arranges the RGB images in a single RGB image and saves the single RGB image.
    
    Parameters
    ----------
    images_uint8 : numpy.ndarray
        4D array with data-type `numpy.uint8`.
        RGB images. `images_uint8[:, :, :, i]` is
        the ith RGB image.
    nb_vertically : int
        Number of RGB images per column
        in the single RGB image.
    path : str
        Path to the saved single RGB image.
        The path must end with ".png".
    
    Raises
    ------
    TypeError
        If `images_uint8.dtype` is not equal to `numpy.uint8`.
    ValueError
        If `images_uint8.shape[2]` is not equal to 3.
    ValueError
        If `images_uint8.shape[3]` is not
        divisible by `nb_vertically`.
    
    """
    if images_uint8.dtype != numpy.uint8:
        raise TypeError('`images_uint8.dtype` is not equal to `numpy.uint8`.')
    (height_image, width_image, nb_channels, nb_images) = images_uint8.shape
    if nb_channels != 3:
        raise ValueError('`images_uint8.shape[2]` is not equal to 3.')
    if nb_images % nb_vertically != 0:
        raise ValueError('`images_uint8.shape[3]` is not divisible by `nb_vertically`.')
    
    # `nb_horizontally` has to be an integer.
    nb_horizontally = nb_images//nb_vertically
    image_uint8 = 255*numpy.ones((nb_vertically*(height_image + 1) + 1,
        nb_horizontally*(width_image + 1) + 1, 3), dtype=numpy.uint8)
    for i in range(nb_vertically):
        for j in range(nb_horizontally):
            image_uint8[i*(height_image + 1) + 1:(i + 1)*(height_image + 1),
                j*(width_image + 1) + 1:(j + 1)*(width_image + 1), :] = \
                images_uint8[:, :, :, i*nb_horizontally + j]
    save_image(path,
               image_uint8)

def visualize_rows(rows_uint8, height_image, width_image, nb_vertically, path):
    """Reshapes each row to a RGB image, arranges the RGB images in a single RGB image and saves the single RGB image.
    
    Parameters
    ----------
    rows_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        `rows_uint8[i, :]` contains the ith RGB image.
    height_image : int
        RGB image height.
    width_image : int
        RGB image width.
    nb_vertically : int
        Number of RGB images per column
        in the single RGB image.
    path : str
        Path to the saved single RGB image.
        The path must end with ".png".
    
    """
    images_uint8 = rows_to_images(rows_uint8,
                                  height_image,
                                  width_image)
    visualize_images(images_uint8,
                     nb_vertically,
                     path)

def visualize_weights(weights, height_image, width_image, nb_vertically, path):
    """Arranges the weight filters in a single RGB image and saves the single RGB image.
    
    Parameters
    ----------
    weights : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Weights filters. `weights[i, :]` contains
        the ith weight filter.
    height_image : int
        Image height.
    width_image : int
        Image width.
    nb_vertically : int
        Number of weight filters per column
        in the single RGB image.
    path : str
        Path to the saved single RGB image.
        The path must end with ".png".
    
    """
    min_w = numpy.amin(weights)
    max_w = numpy.amax(weights)
    rows_uint8 = numpy.round(255.*(weights - min_w)/(max_w - min_w)).astype(numpy.uint8)
    visualize_rows(rows_uint8,
                   height_image,
                   width_image,
                   nb_vertically,
                   path)


