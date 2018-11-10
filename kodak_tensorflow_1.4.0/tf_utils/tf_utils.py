"""A library that contains Tensorflow utilities."""

import tensorflow as tf

# The functions are sorted in
# alphabetic order.

def add_noise(data, bin_widths):
    """Adds zero-mean uniform noise to the data.
    
    Parameters
    ----------
    data : Tensor
        4D tensor with data-type `tf.float32`.
        Data.
    bin_widths : Tensor
        1D tensor with data-type `tf.float32`.
        Quantization bin widths. `bin_widths[i]`
        corresponds to the length of the support
        of the zero-mean uniform noise that is
        added to `tf.slice(data, [0, 0, 0, i], [-1, -1, -1, 1])`.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Data with additive zero-mean uniform noise.
    
    Raises
    ------
    AssertionError
        If a quantization bin width is not
        strictly positive.
    
    """
    # The shape of `data` does not change
    # while running the graph. Therefore,
    # the static shape of `data` is used.
    shape_data = data.get_shape().as_list()
    with tf.control_dependencies([tf.assert_positive(bin_widths)]):
        tiled_bin_widths = tf.tile(tf.reshape(bin_widths, [1, 1, 1, shape_data[3]]),
                                   [shape_data[0], shape_data[1], shape_data[2], 1])
    return data + tiled_bin_widths*tf.random_uniform(shape_data, minval=-0.5, maxval=0.5, dtype=tf.float32)

def approximate_entropy(approximate_prob, bin_widths):
    """Computes the approximate cumulated entropy of the quantized latent variables.
    
    The differential entropy of the ith set of latent
    variables perturbed by uniform noise is first computed
    using the ith piecewise linear function. Then, the
    differential entropy of the ith set of latent variables
    perturbed by uniform noise is converted into the approximate
    entropy of the ith set of quantized latent variables using
    the ith quantization bin width. Finally, the approximate
    entropy is summed over all sets.
    
    Parameters
    ----------
    approximate_prob : Tensor
        2D tensor with data-type `tf.float32`.
        Approximate probabilities. `tf.slice(approximate_prob, [i, 0], [1, -1])`
        contains the approximate probabilities for
        the ith set of latent variables perturbed
        by uniform noise.
    bin_widths : Tensor
        1D tensor with data-type `tf.float32`.
        Quantization bin widths.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Approximate cumulated entropy of the quantized
        latent variables.
    
    Raises
    ------
    AssertionError
        If a quantization bin width is not
        strictly positive.
    AssertionError
        If the approximate entropy of a set of quantized
        latent variables is not positive.
    
    """
    # If a tensor containing negative elements is
    # inserted into the function `tf.log`, no exception
    # is raised. The assertion below corrects this.
    with tf.control_dependencies([tf.assert_positive(bin_widths)]):
        diff_entropies = differential_entropy(approximate_prob)
        approx_entropies = diff_entropies - tf.log(bin_widths)/tf.log(2.)
    with tf.control_dependencies([tf.assert_non_negative(approx_entropies)]):
        return tf.reduce_sum(approx_entropies)

def approximate_probability(samples, parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the approximate probability of each sample.
    
    Parameters
    ----------
    samples : Tensor
        2D tensor with data-type `tf.float32`.
        Samples from unknown probability density functions.
        `tf.slice(samples, [i, 0], [1, -1])` contains the samples
        from the ith unknown probability density function.
    parameters : Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions.
        `tf.slice(parameters, [i, 0], [1, -1])` contains
        the parameters of the ith piecewise linear function.
        The ith piecewise linear function approximates the
        ith unknown probability density function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : Tensor
        0D tensor with data-type `tf.int64`.
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    Tensor
        2D tensor with data-type `tf.float32`.
        Approximate probabilities. The element at the
        position [i, j] in this tensor is the approximate
        probability of `tf.slice(samples, [i, j], [1, 1])`.
    
    """
    # `idx_linear_piece.dtype` is equal to `tf.int64`.
    idx_linear_piece = index_linear_piece(samples,
                                          nb_points_per_interval,
                                          nb_intervals_per_side)
    
    # The shape of `samples` does not change
    # while running the graph. Therefore,
    # the static shape of `samples` is used.
    shape_data = samples.get_shape().as_list()
    nb_maps = shape_data[0]
    nb_elements_per_map = shape_data[1]
    nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
    sequence = nb_points*tf.reshape(tf.cast(tf.linspace(0., tf.cast(nb_maps - 1, tf.float32), nb_maps), tf.int64),
                                    [nb_maps, 1])
    idx_linear_piece_1d = tf.reshape(idx_linear_piece + tf.tile(sequence, [1, nb_elements_per_map]), [-1])
    parameters_1d = tf.reshape(parameters, [-1])
    
    # The 1st argument of the function `tf.gather`
    # must be a Tensorflow variable.
    right = tf.reshape(tf.gather(parameters_1d, idx_linear_piece_1d + 1),
                       [nb_maps, nb_elements_per_map])
    left = tf.reshape(tf.gather(parameters_1d, idx_linear_piece_1d),
                      [nb_maps, nb_elements_per_map])
    left_bound = tf.floor(nb_points_per_interval*samples)/nb_points_per_interval
    return (right - left)*(samples - left_bound)*nb_points_per_interval + left

def area_under_piecewise_linear_functions(parameters, nb_points_per_interval, nb_intervals_per_side):
    """Computes the area under each piecewise linear function.
    
    Parameters
    ----------
    parameters : Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions.
        `tf.slice(parameters, [i, 0], [1, -1])` contains
        the parameters of the ith piecewise linear function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : Tensor
        0D tensor with data-type `tf.int64`.
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    Tensor
        1D tensor with data-type `tf.float32`.
        Its ith element is the area under the
        ith piecewise linear function.
    
    """
    # The shape of `parameters` may change
    # while running the graph. Therefore, the
    # dynamic shape of `parameters` is used.
    nb_maps = tf.shape(parameters)[0]
    nb_points = tf.cast(2*nb_points_per_interval*nb_intervals_per_side + 1, tf.int32)
    return 0.5*(tf.reshape(tf.slice(parameters, [0, 0], [nb_maps, 1]) + tf.slice(parameters, [0, nb_points - 1], [nb_maps, 1]), [-1]) +
        2.*tf.reduce_sum(tf.slice(parameters, [0, 1], [nb_maps, nb_points - 2]), axis=1))/nb_points_per_interval

def differential_entropy(approximate_prob):
    """Computes the differential entropy of each set of samples.
    
    Parameters
    ----------
    approximate_prob : Tensor
        2D tensor with data-type `tf.float32`.
        Approximate probabilities. `tf.slice(approximate_prob, [i, 0], [1, -1])`
        contains the approximate probabilities for
        the ith set of samples.
    
    Returns
    -------
    Tensor
        1D tensor with data-type `tf.float32`.
        Differential entropies. Its ith element is the
        differential entropy of the ith set of samples.
    
    """
    return -tf.reduce_mean(tf.log(approximate_prob)/tf.log(2.), axis=1)

def expand_all(grid, parameters, low_projection, nb_points_per_interval, nb_intervals_per_side, max_abs):
    """Expands the grid and the parameters of the piecewise linear functions if the condition of expansion is met.
    
    Parameters
    ----------
    grid : Tensor
        1D tensor with data-type `tf.float32`.
        Grid storing the sampling points.
    parameters : Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions.
        `tf.slice(parameters, [i, 0], [1, -1])` contains
        the parameters of the ith piecewise linear function.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear functions. Thanks to
        `low_projection`, the parameters of the piecewise
        linear functions cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : Tensor
        0D tensor with data-type `tf.int64`.
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    max_abs : Tensor
        0D tensor with data-type `tf.float32`.
        Largest absolute latent variable plus half the
        largest quantization bin width. The condition
        of expansion is met when `max_abs` is larger
        than `nb_intervals_per_side`.
    
    Returns
    -------
    tuple
        Tensor
            1D tensor with data-type `tf.float32`.
            Grid after the expansion.
        Tensor
            2D tensor with data-type `tf.float32`.
            Parameters of the piecewise linear functions
            after the expansion.
        Tensor
            0D tensor with data-type `tf.int64`.
            Number of unit intervals in the right half of
            the grid after the expansion.
    
    """
    is_expansion = tf.greater_equal(max_abs, tf.cast(nb_intervals_per_side, tf.float32))
    
    # If the above condition is an equality,
    # the grid and the parameters of the piecewise
    # linear functions must be expanded. That is
    # why 1 is added.
    nb_added_per_side = tf.cast(tf.ceil(max_abs), tf.int64) - nb_intervals_per_side + 1
    nb_intervals_per_side_exp = tf.cond(is_expansion,
                                        lambda: nb_intervals_per_side + nb_added_per_side,
                                        lambda: tf.identity(nb_intervals_per_side))
    
    # The data type of the tensor the function
    # `tf.linspace` returns is the same as the
    # data type of the tensor that is passed as
    # its 1st argument.
    grid_exp = tf.cond(is_expansion,
                       lambda: tf.linspace(-tf.cast(nb_intervals_per_side_exp, tf.float32),
                                           tf.cast(nb_intervals_per_side_exp, tf.float32),
                                           tf.cast(2*nb_intervals_per_side_exp*nb_points_per_interval + 1, tf.int32)),
                       lambda: tf.identity(grid))
    parameters_exp = tf.cond(is_expansion,
                             lambda: expand_parameters(parameters,
                                                       low_projection,
                                                       nb_points_per_interval,
                                                       nb_added_per_side),
                             lambda: tf.identity(parameters))
    return (grid_exp, parameters_exp, nb_intervals_per_side_exp)

def expand_parameters(parameters, low_projection, nb_points_per_interval, nb_added_per_side):
    """Expands the parameters of the piecewise linear functions.
    
    Parameters
    ----------
    parameters : Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions.
        `tf.slice(parameters, [i, 0], [1, -1])` contains
        the parameters of the ith piecewise linear function.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear functions. Thanks to
        `low_projection`, the parameters of the piecewise
        linear functions cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_added_per_side : Tensor
        0D tensor with data-type `tf.int64`.
        Number of unit intervals added to each side
        of the grid.
    
    Returns
    -------
    Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions
        after the expansion.
    
    Raises
    ------
    AssertionError
        If `low_projection` is not strictly larger than 1.e-7.
    AssertionError
        If `nb_added_per_side` is not strictly positive.
    
    """
    assert low_projection > 1.e-7, '`low_projection` is not strictly larger than 1.e-7.'
    
    # The shape of `parameters` may change
    # while running the graph. Therefore, the
    # dynamic shape of `parameters` is used.
    nb_maps = tf.shape(parameters)[0]
    
    # The function `tf.assert_positive` checks
    # that each element in its argument tensor
    # is strictly positive.
    with tf.control_dependencies([tf.assert_positive(nb_added_per_side)]):
        piece = low_projection*tf.ones([nb_maps, nb_points_per_interval*tf.cast(nb_added_per_side, tf.int32)],
                                       dtype=tf.float32)
        return tf.concat([piece, parameters, piece], 1)

def gdn(input, gamma, beta):
    """Computes the Generalized Divisive Normalization (GDN).
    
    GDN is described in the paper "Density modeling
    of images using a generalized normalization
    transformation", written by Johannes Balle,
    Valero Laparra and Eero P. Simoncelli (ICLR 2016).
    
    Parameters
    ----------
    input : Tensor
        4D tensor with data-type `tf.float32`.
        Input to GDN.
    gamma : Tensor
        2D tensor with data-type `tf.float32`.
        Weights of GDN.
    beta : Tensor
        1D tensor with data-type `tf.float32`.
        Additive coefficients of GDN.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output from GDN.
    
    """
    # The shape of `input` does not change
    # while running the graph. Therefore,
    # the static shape of `input` is used.
    shape_input = input.get_shape().as_list()
    reshaped_input = tf.reshape(input, [shape_input[0]*shape_input[1]*shape_input[2], shape_input[3]])
    reshaped_output = reshaped_input/tf.sqrt(tf.matmul(reshaped_input**2, gamma) + 
        tf.tile(tf.reshape(beta, [1, shape_input[3]]), [shape_input[0]*shape_input[1]*shape_input[2], 1]))
    return tf.reshape(reshaped_output, shape_input)

def index_linear_piece(samples, nb_points_per_interval, nb_intervals_per_side):
    """Finds the linear piece index of each sample.
    
    Parameters
    ----------
    samples : Tensor
        2D tensor with data-type `tf.float32`.
        Samples from unknown probability density functions.
        `tf.slice(samples, [i, 0], [1, -1])` contains the samples
        from the ith unknown probability density function. The
        ith unknown probability density function is approximated
        by the ith piecewise linear function.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : Tensor
        0D tensor with data-type `tf.int64`.
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    
    Returns
    -------
    Tensor
        2D tensor with data-type `tf.int64`.
        Linear piece indices. The element at the position
        [i, j] in this tensor is the linear piece index of
        `tf.slice(samples, [i, j], [1, 1])`.
    
    Raises
    ------
    AssertionError
        If a linear piece index is not positive.
    AssertionError
        If a linear piece index exceeds the maximum
        possible linear piece index.
    
    """
    idx_linear_piece = tf.cast(tf.floor(nb_points_per_interval*samples), tf.int64) + \
        nb_points_per_interval*nb_intervals_per_side
    max_idx = 2*nb_points_per_interval*nb_intervals_per_side - 1
    
    # The function `tf.assert_non_negative` checks that
    # each element in its argument tensor is positive.
    with tf.control_dependencies([tf.assert_non_negative(idx_linear_piece), tf.assert_less_equal(idx_linear_piece, max_idx)]):
        return tf.identity(idx_linear_piece)

def initialize_weights_gdn(nb_maps, min_gamma):
    """Initializes the weights of GDN/IGDN.
    
    Parameters
    ----------
    nb_maps : int
        Number of feature maps in the input
        to GDN/IGDN. The input to GDN/IGDN
        and the output from GDN/IGDN have
        the same number of feature maps.
    min_gamma : float
        Minimum for the weights of GDN/IGDN.
        `min_gamma` must belong to ]0., 0.01].
    
    Returns
    -------
    Tensor
        2D tensor data-type `tf.float32`.
        Weights of GDN/IGDN. This tensor is
        a symmetric matrix.
    
    Raises
    ------
    AssertionError
        If `min_gamma` does not belong to ]0., 0.01].
    
    """
    assert min_gamma <= 0.01 and min_gamma > 0., '`min_gamma` does not belong to ]0., 0.01].'
    gamma_non_symmetric = tf.random_uniform([nb_maps, nb_maps],
                                            minval=min_gamma,
                                            maxval=0.01,
                                            dtype=tf.float32)
    return 0.5*(gamma_non_symmetric + tf.transpose(gamma_non_symmetric))

def inverse_gdn(input, gamma, beta):
    """Computes the Inverse Generalized Divisive Normalization (IGDN).
    
    Parameters
    ----------
    input : Tensor
        4D tensor with data-type `tf.float32`.
        Input to IGDN.
    gamma : Tensor
        2D tensor with data-type `tf.float32`.
        Weights of IGDN.
    beta : Tensor
        1D tensor with data-type `tf.float32`.
        Additive coefficients of IGDN.
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Output from IGDN.
    
    """
    # The shape of `input` does not change
    # while running the graph. Therefore,
    # the static shape of `input` is used.
    shape_input = input.get_shape().as_list()
    reshaped_input = tf.reshape(input, [shape_input[0]*shape_input[1]*shape_input[2], shape_input[3]])
    reshaped_output = reshaped_input*tf.sqrt(tf.matmul(reshaped_input**2, gamma) +
        tf.tile(tf.reshape(beta, [1, shape_input[3]]), [shape_input[0]*shape_input[1]*shape_input[2], 1]))
    return tf.reshape(reshaped_output, shape_input)

def loss_density_approximation(approximate_prob, parameters, nb_points_per_interval):
    """Computes the loss of the approximation of unknown probability density functions with piecewise linear functions.
    
    Each loss portion is derived from the mean integrated
    squared error between a piecewise linear function and
    the unknown probability density function it approximates.
    The loss is the sum of all the loss portions.
    
    Parameters
    ----------
    approximate_prob : Tensor
        2D tensor with data-type `tf.float32`.
        Approximate probabilities. `tf.slice(approximate_prob, [i, 0], [1, -1])`
        contains the approximate probabilities for
        the ith set of samples.
    parameters : Tensor
        2D tensor with data-type `tf.float32`.
        Parameters of the piecewise linear functions.
        `tf.slice(parameters, [i, 0], [1, -1])` contains
        the parameters of the ith piecewise linear function.
        The ith piecewise linear function approximates the
        probability density function of the ith set of samples.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Loss of the approximation of unknown probability
        density functions with piecewise linear functions.
    
    """
    return tf.reduce_sum(-2.*tf.reduce_mean(approximate_prob, axis=1) + tf.reduce_sum(parameters**2, axis=1)/nb_points_per_interval)

def reconstruction_error(visible_units, reconstruction):
    """Computes the error between the visible units and their reconstruction.
    
    Parameters
    ----------
    visible_units : Tensor
        4D tensor with data-type `tf.float32`.
        Visible units.
    reconstruction : Tensor
        4D tensor with data-type `tf.float32`.
        Reconstruction of the visible units.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Error between the visible units and
        their reconstruction.
    
    """
    return tf.reduce_mean(tf.reduce_sum((visible_units - reconstruction)**2, axis=[1, 2, 3]))

def reshape_4d_to_2d(tensor_4d):
    """Reshapes the 4D tensor into a 2D tensor.
    
    Parameters
    ----------
    tensor_4d : Tensor
        4D tensor.
        Tensor to be reshaped.
        `tf.slice(tensor_4d, [0, 0, 0, i], [-1, -1, -1, 1])`
        is reshaped into the ith row of the tensor resulting
        from the change of shape.
    
    Returns
    -------
    Tensor
        2D tensor.
        Tensor resulting from the
        change of shape.
    
    """
    # The shape of `tensor_4d` does not change
    # while running the graph. Therefore,
    # the static shape of `tensor_4d` is used.
    shape_4d = tensor_4d.get_shape().as_list()
    return tf.transpose(tf.reshape(tensor_4d, [shape_4d[0]*shape_4d[1]*shape_4d[2], shape_4d[3]]))


