"""A script to test the library that contains Tensorflow utilities."""

import argparse
import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy.stats.distributions
import tensorflow as tf

import tf_utils.tf_utils as tfuls
import tools.tools as tls


class TesterTfUtils(object):
    """Class for testing the library that contains Tensorflow utilities."""
    
    def test_add_noise(self):
        """Tests the function `add_noise`.
        
        The test is successful if a tiny zero-mean uniform
        noise is added to the 1st slice of `data` whereas a
        tremendous zero-mean uniform noise is added to the
        2nd slice of `data`.
        
        """
        data = numpy.random.normal(loc=0., scale=1., size=(2, 1, 3, 2)).astype(numpy.float32)
        bin_widths_np = numpy.array([0.01, 100.], dtype=numpy.float32)
        bin_widths = tf.constant(bin_widths_np, dtype=tf.float32)
        node_data = tf.placeholder(tf.float32, shape=(2, 1, 3, 2))
        node_data_tilde = tfuls.add_noise(node_data, bin_widths)
        with tf.Session() as sess:
            print('1st slice of `data` after being flattened:')
            print(data[:, :, :, 0].flatten())
            print('2nd slice of `data` after being flattened:')
            print(data[:, :, :, 1].flatten())
            print('1st quantization bin width: {}'.format(bin_widths_np[0]))
            print('2nd quantization bin width: {}'.format(bin_widths_np[1]))
            data_tilde = sess.run(node_data_tilde, feed_dict={node_data:data})
            print('1st slice of `data_tilde` after being flattened:')
            print(data_tilde[:, :, :, 0].flatten())
            print('2nd slice of `data_tilde` after being flattened:')
            print(data_tilde[:, :, :, 1].flatten())
    
    def test_approximate_entropy(self):
        """Tests the function `approximate_entropy`.
        
        The test is successful if the approximate cumulated
        entropy of the quantized latent variables is close to
        the cumulated entropy of the quantized latent variables.
        
        """
        nb_points_per_interval = 10
        nb_itvs_per_side = 8
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        nb_elements_per_map = 80000
        bin_widths_np = numpy.array([0.5, 0.1, 0.3, 0.05, 1.], dtype=numpy.float32)
        nb_maps = bin_widths_np.size
        bin_widths = tf.constant(bin_widths_np, dtype=tf.float32)
        low_projection = 1.e-6
        nb_epochs_fitting = 120
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        
        # Here, the latent variables `y` are samples
        # from the probability density function of the
        # standard normal distribution.
        y = numpy.random.normal(loc=0.,
                                scale=1.,
                                size=(nb_maps, nb_elements_per_map)).astype(numpy.float32)
        tiled_bin_widths = numpy.tile(numpy.expand_dims(bin_widths_np, axis=1),
                                      (1, nb_elements_per_map))
        samples_standard = numpy.random.uniform(low=-0.5,
                                                high=0.5,
                                                size=(nb_maps, nb_elements_per_map)).astype(numpy.float32)
        samples_uniform = tiled_bin_widths*samples_standard
        y_tilde = y + samples_uniform
        
        # `parameters[i, :]` contains the parameters of the
        # ith piecewise linear function. The ith piecewise
        # linear function will be trained to approximate
        # the probability density function of `y_tilde[i, :]`.
        # Note that the probability density function of `y_tilde[i, :]`
        # is the convolution between the probability density function
        # of the standard normal distribution and the probability
        # density function of the uniform distribution of support
        # [-0.5*`bin_widths_np[i]`, 0.5*`bin_widths_np[i]`].
        parameters = tf.Variable(tls.tile_cauchy(grid, nb_maps),
                                 dtype=tf.float32)
        node_y_tilde = tf.placeholder(tf.float32, shape=(nb_maps, nb_elements_per_map))
        node_approximate_prob = tfuls.approximate_probability(node_y_tilde,
                                                              parameters,
                                                              nb_points_per_interval,
                                                              nb_intervals_per_side)
        node_approx_entropy = tfuls.approximate_entropy(node_approximate_prob,
                                                        bin_widths)
        node_loss_density_approx = tfuls.loss_density_approximation(node_approximate_prob,
                                                                    parameters,
                                                                    nb_points_per_interval)
        node_opt_fct = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(
            node_loss_density_approx,
            var_list=[parameters]
        )
        node_projection_parameters_fct = tf.assign(
            parameters,
            tf.maximum(parameters, low_projection)
        )
        with tf.Session() as sess:
            
            # For details on the condition below, see
            # <https://www.tensorflow.org/api_guides/python/upgrade>.
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            for i in range(nb_epochs_fitting):
                _ = sess.run(node_opt_fct, feed_dict={node_y_tilde:y_tilde})
                _ = sess.run(node_projection_parameters_fct)
            approx_entropy = sess.run(node_approx_entropy, feed_dict={node_y_tilde:y_tilde})
        cumulated_entropy = 0.
        for i in range(nb_maps):
            quantized_y = bin_widths_np[i]*numpy.round(y[i, :]/bin_widths_np[i])
            cumulated_entropy += tls.discrete_entropy(quantized_y,
                                                      bin_widths_np[i].item())
        print('Approximate cumulated entropy of the quantized latent variables: {}'.format(approx_entropy))
        print('Cumulated entropy of the quantized latent variables: {}'.format(cumulated_entropy))
    
    def test_approximate_probability(self):
        """Tests the function `approximate_probability`.
        
        The test is successful if, for each sample,
        the approximate probability of the sample
        computed by hand is almost equal to the approximate
        probability of the sample computed by the function.
        
        """
        numpy.random.seed(0)
        tf.set_random_seed(0)
        nb_points_per_interval = 4
        nb_itvs_per_side = 2
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        nb_maps = 3
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        
        # In the current test, we do not require the
        # piecewise linear functions to approximate the
        # probability density functions from which `samples`
        # are sampled.
        parameters = tf.Variable(tf.random_uniform([nb_maps, nb_points], minval=0., maxval=1., dtype=tf.float32),
                                 dtype=tf.float32,
                                 trainable=False)
        samples = numpy.random.normal(loc=0.,
                                      scale=0.4,
                                      size=(nb_maps, 4)).astype(numpy.float32)
        node_samples = tf.placeholder(tf.float32, shape=(nb_maps, 4))
        node_approximate_prob = tfuls.approximate_probability(node_samples,
                                                              parameters,
                                                              nb_points_per_interval,
                                                              nb_intervals_per_side)
        approximate_prob_hand = numpy.array([[0.879491, 0.619454, 0.634988, 0.873688],
                                             [0.373638, 0.669101, 0.276398, 0.696577],
                                             [0.889826, 0.824482, 0.857850, 0.494299]], dtype=numpy.float32)
        with tf.Session() as sess:
            
            # For details on the condition below, see
            # <https://www.tensorflow.org/api_guides/python/upgrade>.
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
            print('Number of unit intervals in the right half of the grid: {}'.format(nb_intervals_per_side.eval()))
            print('Grid:')
            print(grid)
            print('Parameters of the piecewise linear functions:')
            print(parameters.eval())
            print('Samples:')
            print(samples)
            approximate_prob_fct = sess.run(node_approximate_prob, feed_dict={node_samples:samples})
            print('Approximate probability of each sample computed by the function:')
            print(approximate_prob_fct)
            print('Approximate probability of each sample computed by hand:')
            print(approximate_prob_hand)
    
    def test_area_under_piecewise_linear_functions(self):
        """Tests the function `area_under_piecewise_linear_functions`.
        
        A 1st plot is saved at
        "tf_utils/pseudo_visualization/area_under_piecewise_linear_functions/piecewise_linear_function_0.png".
        A 2nd plot is saved at
        "tf_utils/pseudo_visualization/area_under_piecewise_linear_functions/piecewise_linear_function_1.png".
        The test is successful if, for each plot, the area
        under the piecewise linear function computed by hand
        is almost equal to the area computed by the function.
        
        """
        nb_points_per_interval = 20
        nb_itvs_per_side = 4
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        parameters = -0.5*numpy.ones((2, nb_points))
        parameters[0, 0:nb_points_per_interval*nb_itvs_per_side + 1] = \
            -grid[0:nb_points_per_interval*nb_itvs_per_side + 1] - 0.5
        parameters[1, 0:nb_points_per_interval*nb_itvs_per_side + 1] = \
            -4.*grid[0:nb_points_per_interval*nb_itvs_per_side + 1] - 0.5
        node_parameters = tf.placeholder(tf.float32, shape=(2, nb_points))
        node_area = tfuls.area_under_piecewise_linear_functions(node_parameters,
                                                                nb_points_per_interval,
                                                                nb_intervals_per_side)
        with tf.Session() as sess:
            area = sess.run(node_area, feed_dict={node_parameters:parameters})
        print('Area under the 1st piecewise linear function computed by the function: {}'.format(area[0]))
        print('Area under the 1st piecewise linear function computed by hand: {}'.format(4.0))
        print('Area under the 2nd piecewise linear function computed by the function: {}'.format(area[1]))
        print('Area under the 2nd piecewise linear function computed by hand: {}'.format(28.0))
        plt.plot(grid, parameters[0, :], color='blue')
        plt.title('1st piecewise linear function of integral {}'.format(area[0]))
        plt.savefig('tf_utils/pseudo_visualization/area_under_piecewise_linear_functions/piecewise_linear_function_0.png')
        plt.clf()
        plt.plot(grid, parameters[1, :], color='blue')
        plt.title('2nd piecewise linear function of integral {}'.format(area[1]))
        plt.savefig('tf_utils/pseudo_visualization/area_under_piecewise_linear_functions/piecewise_linear_function_1.png')
        plt.clf()
    
    def test_differential_entropy(self):
        """Tests the function `differential_entropy`.
        
        The test is successful if the theoretical
        differential entropy is almost equal to the
        4 differential entropies computed by the function.
        
        """
        nb_points_per_interval = 10
        nb_itvs_per_side = 8
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        nb_elements_per_map = 20000
        nb_maps = 4
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        pdf_float32 = scipy.stats.distributions.norm.pdf(grid, loc=0., scale=1.).astype(numpy.float32)
        parameters = tf.Variable(numpy.tile(pdf_float32, (nb_maps, 1)),
                                 dtype=tf.float32,
                                 trainable=False)
        
        # The samples are drawn from the standard normal distribution
        # as the probability density function of the standard normal
        # distribution is chosen to create the parameters of each
        # piecewise linear function.
        samples = numpy.random.normal(loc=0.,
                                      scale=1.,
                                      size=(nb_maps, nb_elements_per_map)).astype(numpy.float32)
        node_samples = tf.placeholder(tf.float32, shape=(nb_maps, nb_elements_per_map))
        node_approximate_prob = tfuls.approximate_probability(node_samples,
                                                              parameters,
                                                              nb_points_per_interval,
                                                              nb_intervals_per_side)
        node_differential_entropies = tfuls.differential_entropy(node_approximate_prob)
        with tf.Session() as sess:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            diff_entropies = sess.run(node_differential_entropies, feed_dict={node_samples:samples})
        theoretical_diff_entropy = 0.5*(1. + numpy.log(2.*numpy.pi))/numpy.log(2.)
        print('Theoretical differential entropy: {} bits.'.format(theoretical_diff_entropy))
        print('1st differential entropy computed by the function: {} bits.'.format(diff_entropies[0]))
        print('2nd differential entropy computed by the function: {} bits.'.format(diff_entropies[1]))
        print('3rd differential entropy computed by the function: {} bits.'.format(diff_entropies[2]))
        print('4th differential entropy computed by the function: {} bits.'.format(diff_entropies[3]))
    
    def test_expand_all(self):
        """Tests the function `expand_all`.
        
        The test is successful if the expansion
        adds one unit interval to the right half
        of the grid and one unit interval to the
        left half of the grid.
        
        """
        nb_points_per_interval = 4
        nb_itvs_per_side = 2
        nb_intervals_per_side = tf.Variable(nb_itvs_per_side,
                                            dtype=tf.int64,
                                            trainable=False)
        low_projection = 1.e-6
        max_abs = tf.constant(2.)
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        
        # `grid_init` is used as initializer.
        # An initializer must have the same
        # data-type as the tensor it initializes.
        grid_init = numpy.linspace(-nb_itvs_per_side,
                                   nb_itvs_per_side,
                                   num=nb_points,
                                   dtype=numpy.float32)
        grid = tf.Variable(grid_init,
                           dtype=tf.float32,
                           trainable=False)
        parameters = tf.Variable(tf.random_uniform([2, nb_points], minval=0., maxval=1., dtype=tf.float32),
                                 dtype=tf.float32,
                                 trainable=False)
        (grid_exp, parameters_exp, nb_intervals_per_side_exp) = tfuls.expand_all(grid,
                                                                                 parameters,
                                                                                 low_projection,
                                                                                 nb_points_per_interval,
                                                                                 nb_intervals_per_side,
                                                                                 max_abs)
        node_expansion = [
            tf.assign(grid, grid_exp, validate_shape=False),
            tf.assign(parameters, parameters_exp, validate_shape=False),
            tf.assign(nb_intervals_per_side, nb_intervals_per_side_exp)
        ]
        with tf.Session() as sess:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
            print('Number of unit intervals in the right half of the grid before the expansion: {}'.format(nb_intervals_per_side.eval()))
            print('Largest absolute latent variable plus half the largest quantization bin width: {}'.format(max_abs.eval()))
            print('Grid before the expansion:')
            print(grid.eval())
            print('Parameters of the piecewise linear functions before the expansion:')
            print(parameters.eval())
            _ = sess.run(node_expansion)
            print('Number of unit intervals in the right half of the grid after the expansion: {}'.format(nb_intervals_per_side.eval()))
            print('Grid after the expansion:')
            print(grid.eval())
            print('Parameters of the piecewise linear functions after the expansion:')
            print(parameters.eval())
    
    def test_expand_parameters(self):
        """Tests the function `expand_parameters`.
        
        The test is successful if the expansion
        adds 4 columns to the left side of the
        matrix storing the parameters of the
        piecewise linear functions and 4 columns
        to its right side.
        
        """
        nb_points_per_interval = 2
        nb_itvs_per_side = 2
        low_projection = 1.e-6
        nb_added_per_side = tf.constant(2, dtype=tf.int64)
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        parameters_uniform = numpy.random.uniform(low=0., high=0.1, size=(1, nb_points)).astype(numpy.float32)
        parameters_normal = numpy.random.normal(loc=0., scale=4., size=(1, nb_points)).astype(numpy.float32)
        parameters = tf.Variable(numpy.concatenate((parameters_uniform, parameters_normal), axis=0),
                                 dtype=tf.float32,
                                 trainable=False)
        parameters_exp = tfuls.expand_parameters(parameters,
                                                 low_projection,
                                                 nb_points_per_interval,
                                                 nb_added_per_side)
        node_expansion = tf.assign(parameters,
                                   parameters_exp,
                                   validate_shape=False)
        with tf.Session() as sess:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
            print('Number of unit intervals added to each side of the grid: {}'.format(nb_added_per_side.eval()))
            print('Parameters of the piecewise linear functions before the expansion:')
            print(parameters.eval())
            _ = sess.run(node_expansion)
            print('Parameters of the piecewise linear functions after the expansion:')
            print(parameters.eval())
    
    def test_gdn(self):
        """Tests the function `gdn`.
        
        The test is successful if the output from
        GDN is equal to the input to GDN divided
        by the square root of the common value for
        the additive coefficients of GDN.
        
        """
        shape_input = (2, 2, 2, 2)
        
        input = numpy.random.normal(loc=0., scale=1., size=shape_input).astype(numpy.float32)
        gamma = tf.zeros([shape_input[3], shape_input[3]], dtype=tf.float32)
        beta = 4.*tf.ones([shape_input[3]], dtype=tf.float32)
        node_input = tf.placeholder(tf.float32, shape=shape_input)
        node_output = tfuls.gdn(node_input, gamma, beta)
        with tf.Session() as sess:
            output = sess.run(node_output, feed_dict={node_input:input})
            print('Weights of GDN:')
            print(gamma.eval())
            print('Additive coefficients of GDN:')
            print(beta.eval())
        print('Input to GDN:')
        print(input)
        print('Output from GDN:')
        print(output)
    
    def test_index_linear_piece(self):
        """Tests the function `index_linear_piece`.
        
        The test is successful if, for each sample,
        the linear piece index of the sample computed
        by the function is equal to the linear piece
        index computed by hand.
        
        """
        nb_points_per_interval = 4
        nb_itvs_per_side = 3
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        
        # The index of the last point of the
        # grid is the linear piece index of
        # samples that are out of the grid
        # bounds. This index does not appear as
        # any "out" sample raises an exception.
        samples = numpy.array([[-3., -0.01, 0., 0.01], [-1.13, 2.99, 2., -0.47]],
                              dtype=numpy.float32)
        node_samples = tf.placeholder(tf.float32, shape=(2, 4))
        node_idx_linear_piece = tfuls.index_linear_piece(node_samples,
                                                         nb_points_per_interval,
                                                         nb_intervals_per_side)
        with tf.Session() as sess:
            print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
            print('Number of unit intervals in the right half of the grid: {}'.format(nb_intervals_per_side.eval()))
            print('Grid:')
            print(grid)
            print('Samples:')
            print(samples)
            idx_linear_piece = sess.run(node_idx_linear_piece, feed_dict={node_samples:samples})
        print('Linear piece index of each sample computed by the function:')
        print(idx_linear_piece)
        print('Linear piece index of each sample computed by hand:')
        print(numpy.array([[0, 11, 12, 12], [7, 23, 20, 10]], dtype=numpy.int64))
    
    def test_initialize_weights_gdn(self):
        """Tests the function `initialize_weights_gdn`.
        
        An image is saved at
        "tf_utils/pseudo_visualization/initialize_weights_gdn.png".
        The test is successful if the image
        contains a symmetric matrix. Besides,
        the GDN/IGDN weights must belong to the
        expected range.
        
        """
        nb_maps = 12
        min_gamma = 2.e-5
        
        node_gamma = tfuls.initialize_weights_gdn(nb_maps, min_gamma)
        with tf.Session() as sess:
            gamma = sess.run(node_gamma)
        minimum = numpy.amin(gamma)
        maximum = numpy.amax(gamma)
        print('The GDN/IGDN weights must belong to [{}, 0.01].'.format(min_gamma))
        print('Minimum of the GDN/IGDN weights: {}'.format(minimum))
        print('Maximum of the GDN/IGDN weights: {}'.format(maximum))
        image_uint8 = numpy.round(255.*(gamma - minimum)/(maximum - minimum)).astype(numpy.uint8)
        tls.save_image('tf_utils/pseudo_visualization/initialize_weights_gdn.png',
                       image_uint8)
    
    def test_inverse_gdn(self):
        """Tests the function `inverse_gdn`.
        
        The test is successful if the output from
        IGDN is equal to the input to IGDN multiplied
        by the square root of the common value for the
        additive coefficients of IGDN.
        
        """
        shape_input = (2, 2, 2, 2)
        
        input = numpy.random.normal(loc=0., scale=1., size=shape_input).astype(numpy.float32)
        gamma = tf.zeros([shape_input[3], shape_input[3]], dtype=tf.float32)
        beta = 4.*tf.ones([shape_input[3]], dtype=tf.float32)
        node_input = tf.placeholder(tf.float32, shape=shape_input)
        node_output = tfuls.inverse_gdn(node_input, gamma, beta)
        with tf.Session() as sess:
            output = sess.run(node_output, feed_dict={node_input:input})
            print('Weights of IGDN:')
            print(gamma.eval())
            print('Additive coefficients of IGDN:')
            print(beta.eval())
        print('Input to IGDN:')
        print(input)
        print('Output from IGDN:')
        print(output)
    
    def test_loss_density_approximation(self):
        """Tests the function `loss_density_approximation`.
        
        The test is successful if the loss
        computed by the function is almost
        equal to the loss computed by hand.
        
        """
        nb_points_per_interval = 10
        nb_itvs_per_side = 3
        nb_intervals_per_side = tf.constant(nb_itvs_per_side, dtype=tf.int64)
        
        nb_points = 2*nb_points_per_interval*nb_itvs_per_side + 1
        grid = numpy.linspace(-nb_itvs_per_side,
                              nb_itvs_per_side,
                              num=nb_points)
        parameters_0 = numpy.expand_dims(scipy.stats.uniform.pdf(grid, loc=-1., scale=2.).astype(numpy.float32),
                                         axis=0)
        parameters_1 = numpy.expand_dims(scipy.stats.triang.pdf(grid, 0.5, loc=0., scale=2.).astype(numpy.float32),
                                         axis=0)
        parameters = tf.Variable(numpy.concatenate((parameters_0, parameters_1), axis=0),
                                 dtype=tf.float32,
                                 trainable=False)
        samples_0 = numpy.random.uniform(low=-1.,
                                         high=1.,
                                         size=(1, 5000)).astype(numpy.float32)
        samples_1 = numpy.random.triangular(0.,
                                            1.,
                                            2.,
                                            size=(1, 5000)).astype(numpy.float32)
        samples = numpy.concatenate((samples_0, samples_1), axis=0)
        node_samples = tf.placeholder(tf.float32, shape=(2, 5000))
        node_approximate_prob = tfuls.approximate_probability(node_samples,
                                                              parameters,
                                                              nb_points_per_interval,
                                                              nb_intervals_per_side)
        
        # For the piecewise linear function with parameters `parameters_0`,
        # the loss is the sum of two terms. The 1st term must be equal to
        # -1.0. The 2nd term (the integral of the square of the p.d.f of the
        # continuous uniform distribution of support [-1.0, 1.0]) must be
        # equal to 0.5.
        # For the piecewise linear function with parameters `parameters_1`,
        # the loss is the sum of two terms. The 1st term must be equal to
        # -4/3. The 2nd term (the integral of the square of the p.d.f of the
        # triangular distribution with lower limit 0.0, upper limit 2.0 and
        # mode 1.0) must be equal to 2/3.
        # The loss below is the sum of the two previous losses.
        node_loss_density_approx = tfuls.loss_density_approximation(node_approximate_prob,
                                                                    parameters,
                                                                    nb_points_per_interval)
        with tf.Session() as sess:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            loss_density_approx = sess.run(node_loss_density_approx, feed_dict={node_samples:samples})
            print('Loss computed by the function: {}'.format(loss_density_approx))
            print('Loss computed by hand: {}'.format(-1.166667))
    
    def test_reconstruction_error(self):
        """Tests the function `reconstruction_error`.
        
        The test is successful if the reconstruction
        error computed by the function is almost equal
        to the reconstruction error computed by hand.
        
        """
        visible_units = numpy.ones((2, 4, 5, 3))
        visible_units[0, :, :, :] = 2*visible_units[0, :, :, :]
        reconstruction = numpy.ones((2, 4, 5, 3))
        node_visible_units = tf.placeholder(tf.float32, shape=(2, 4, 5, 3))
        node_reconstruction = tf.placeholder(tf.float32, shape=(2, 4, 5, 3))
        node_rec_error = tfuls.reconstruction_error(node_visible_units,
                                                    node_reconstruction)
        with tf.Session() as sess:
            rec_error = sess.run(
                node_rec_error,
                feed_dict={node_visible_units:visible_units, node_reconstruction:reconstruction}
            )
        print('Reconstruction error computed by the function: {}'.format(rec_error))
        print('Reconstruction error computed by hand: {}'.format(30.))
    
    def test_reshape_4d_to_2d(self):
        """Tests the function `reshape_4d_to_2d`.
        
        The test is successful if the 1st slice of
        the 4D tensor contains the same elements as
        the 1st row of the 2D tensor. Besides, the
        2nd slice of the 4D tensor must contain the
        same elements as the 2nd row of the 2D tensor.
        
        """
        example_0_slice_0 = numpy.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                                        dtype=numpy.float32)
        example_0_slice_1 = numpy.array([[-1., -2., -3.], [-4., -5., -6.], [-7., -8., -9.]],
                                        dtype=numpy.float32)
        tuple_slices_0 = (
            numpy.expand_dims(example_0_slice_0, axis=2),
            numpy.expand_dims(example_0_slice_1, axis=2)
        )
        example_0 = numpy.expand_dims(numpy.concatenate(tuple_slices_0, axis=2),
                                      axis=0)
        example_1_slice_0 = numpy.array([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]],
                                        dtype=numpy.float32)
        example_1_slice_1 = numpy.array([[-9., -8., -7.], [-6., -5., -4.], [-3., -2., -1.]],
                                        dtype=numpy.float32)
        tuple_slices_1 = (
            numpy.expand_dims(example_1_slice_0, axis=2),
            numpy.expand_dims(example_1_slice_1, axis=2)
        )
        example_1 = numpy.expand_dims(numpy.concatenate(tuple_slices_1, axis=2),
                                      axis=0)
        tensor_4d = numpy.concatenate((example_0, example_1), axis=0)
        print('1st slice of the 4D tensor:')
        print(tensor_4d[:, :, :, 0])
        print('2nd slice of the 4D tensor:')
        print(tensor_4d[:, :, :, 1])
        node_tensor_4d = tf.placeholder(tf.float32, shape=(2, 3, 3, 2))
        node_tensor_2d = tfuls.reshape_4d_to_2d(node_tensor_4d)
        with tf.Session() as sess:
            tensor_2d = sess.run(node_tensor_2d, feed_dict={node_tensor_4d:tensor_4d})
        print('2D tensor:')
        print(tensor_2d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library that contains Tensorflow utilities.')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterTfUtils()
    getattr(tester, 'test_' + args.name)()


