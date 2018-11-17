"""A script to test the library that contains common tools."""

import argparse
import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy.misc
import scipy.stats.distributions
import warnings

import tools.tools as tls

def fit_piecewise_linear_function(samples, grid, low_projection, nb_points_per_interval,
                                  nb_intervals_per_side, nb_epochs_fitting, learning_rate=0.2):
    """Fits a piecewise linear function to the unknown probability density function.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from the unknown probability density
        function.
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
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
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    nb_epochs_fitting : int
        Number of fitting epochs.
    learning_rate : float, optional
        Learning rate for the parameters of the piecewise
        linear function. The default value is 0.2.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function
        after the fitting.
    
    """
    parameters = numpy.maximum(scipy.stats.distributions.cauchy.pdf(grid),
                               low_projection)
    for _ in range(nb_epochs_fitting):
        gradients = tls.gradient_density_approximation(samples,
                                                       parameters,
                                                       nb_points_per_interval,
                                                       nb_intervals_per_side)
        parameters -= learning_rate*gradients
        parameters = numpy.maximum(parameters,
                                   low_projection)
    return parameters


class TesterTools(object):
    """Class for testing the library that contains common tools."""
    
    def test_approximate_entropy(self):
        """Tests the function `approximate_entropy`.
        
        The test compares two different relations between
        the differential entropy of samples from a probability
        density function and the entropy of the quantized
        samples. The 1st relation is the theorem 8.3.1 in the
        book "Elements of information theory, second edition",
        written by Thomas M. Cover and Joy A. Thomas, page 248.
        Note that, in this theorem, the quantization is defined
        via the mean value theorem whereas, in our case, the
        quantization is scalar uniform. This difference reduces
        the accuracy of the 1st relation. The 2nd relation is
        implemented by the function `approximate_entropy`.
        Three plots are saved in the directory at "tools/pseudo_visualization/approximate_entropy/".
        The test is successful if, in the three plots, for the
        two relations, the error of entropy estimation is low
        at low quantization bin width.
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 30
        bin_widths = numpy.linspace(0.2, 6., num=30)
        low_projection = 1.e-6
        nb_epochs_fitting = 120
        
        # `scale_normal`, `scale_logistic`, and `scale_laplace`
        # are the scales of respectively the normal distribution,
        # the logistic, and the Laplace distribution.
        # distribution.
        scale_normal = 2.
        scale_logistic = 1.
        scale_laplace = 1.
        
        nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        
        def test_approximate_entropy_changing_pdf(y, theoretical_diff_entropy, path):
            gaps = numpy.zeros((2, bin_widths.size))
            for i in range(bin_widths.size):
                bin_width = bin_widths[i].item()
                approx_entropy_0 = theoretical_diff_entropy - numpy.log2(bin_width)
                quantized_y = tls.quantization(y, bin_width)
                disc_entropy = tls.discrete_entropy(quantized_y, bin_width)
                gaps[0, i] = numpy.absolute(disc_entropy - approx_entropy_0)
                samples_uniform = numpy.random.uniform(low=-0.5*bin_width,
                                                       high=0.5*bin_width,
                                                       size=y.size)
                y_tilde = y + samples_uniform
                
                # `parameters` are the parameters of the piecewise
                # linear function. The piecewise linear function
                # approximates the probability density function of
                # `y_tilde`. Note that the probability density function
                # of `y_tilde` is the convolution between the probability
                # density function of `y` and the probability density
                # function of the continuous uniform distribution of
                # support [-0.5*`bin_width`, 0.5*`bin_width`].
                parameters = fit_piecewise_linear_function(y_tilde,
                                                           grid,
                                                           low_projection,
                                                           nb_points_per_interval,
                                                           nb_intervals_per_side,
                                                           nb_epochs_fitting)
                approx_entropy_1 = tls.approximate_entropy(y_tilde,
                                                           parameters,
                                                           nb_points_per_interval,
                                                           nb_intervals_per_side,
                                                           bin_width)
                gaps[1, i] = numpy.absolute(disc_entropy - approx_entropy_1)
            tls.plot_graphs(bin_widths,
                            gaps,
                            'quantization bin width',
                            'error of entropy estimation',
                            ['1st relation', '2nd relation'],
                            ['r', 'b'],
                            'Evolution of the error of entropy estimation \n with the quantization bin width',
                            path)
        
        # `theoretical_diff_entropy_0` is the differential
        # entropy of the probability density function of the
        # normal distribution of scale `scale_normal`.
        y_0 = numpy.random.normal(loc=1.,
                                  scale=scale_normal,
                                  size=60000)
        theoretical_diff_entropy_0 = 0.5*(1. + numpy.log(2.*numpy.pi*scale_normal**2))/numpy.log(2.)
        test_approximate_entropy_changing_pdf(y_0,
                                              theoretical_diff_entropy_0,
                                              'tools/pseudo_visualization/approximate_entropy/approximate_entropy_normal_{}.png'.format(tls.float_to_str(scale_normal)))
        
        # `theoretical_diff_entropy_1` is the differential
        # entropy of the probability density function of the
        # logistic distribution.
        y_1 = numpy.random.logistic(loc=-1.,
                                    scale=scale_logistic,
                                    size=60000)
        theoretical_diff_entropy_1 = 2./numpy.log(2.)
        test_approximate_entropy_changing_pdf(y_1,
                                              theoretical_diff_entropy_1,
                                              'tools/pseudo_visualization/approximate_entropy/approximate_entropy_logistic.png')
        
        # `theoretical_diff_entropy_2` is the differential
        # entropy of the probability density function of the
        # Laplace distribution of scale `scale_laplace`.
        y_2 = numpy.random.laplace(loc=0.,
                                   scale=scale_laplace,
                                   size=60000)
        theoretical_diff_entropy_2 = (1. + numpy.log(2.*scale_laplace))/numpy.log(2.)
        test_approximate_entropy_changing_pdf(y_2,
                                              theoretical_diff_entropy_2,
                                              'tools/pseudo_visualization/approximate_entropy/approximate_entropy_laplace_{}.png'.format(tls.float_to_str(scale_laplace)))
    
    def test_approximate_probability(self):
        """Tests the function `approximate_probability`.
        
        The test is successful if, for each sample,
        the approximate probability of the sample
        computed by hand is almost equal to the
        approximate probability of the sample
        computed by the function.
        
        """
        numpy.random.seed(0)
        nb_points_per_interval = 4
        nb_intervals_per_side = 2
        
        nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        
        # In the current test, we do not require the
        # piecewise linear function to approximate the
        # probability density function from which `samples`
        # is sampled.
        parameters = numpy.random.uniform(low=0.,
                                          high=1.,
                                          size=nb_points)
        samples = numpy.random.normal(loc=0.,
                                      scale=0.4,
                                      size=3)
        print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
        print('Number of unit intervals in the right half of the grid: {}'.format(nb_intervals_per_side))
        print('Grid:')
        print(grid)
        print('Parameters of the piecewise linear function:')
        print(parameters)
        print('Samples:')
        print(samples)
        approximate_prob = tls.approximate_probability(samples,
                                                       parameters,
                                                       nb_points_per_interval,
                                                       nb_intervals_per_side)
        print('Approximate probability of each sample computed by the function:')
        print(approximate_prob)
        print('Approximate probability of each sample computed by hand:')
        print([0.514050, 0.426015, 0.942776])

    def test_area_under_piecewise_linear_function(self):
        """Tests the function `area_under_piecewise_linear_function`.
        
        A plot is saved at
        "tools/pseudo_visualization/area_under_piecewise_linear_function.png".
        The test is successful if the area under
        the piecewise linear function computed by
        the function is almost equal to the area
        under the piecewise linear function computed
        by hand.
        
        """
        nb_points_per_interval = 20
        nb_intervals_per_side = 4
        
        nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = -0.5*numpy.ones(nb_points)
        parameters[0:nb_points_per_interval*nb_intervals_per_side + 1] = \
            -grid[0:nb_points_per_interval*nb_intervals_per_side + 1] - 0.5
        area = tls.area_under_piecewise_linear_function(parameters,
                                                        nb_points_per_interval)
        print('Area under the piecewise linear function computed by the function: {}'.format(area))
        print('Area under the piecewise linear function computed by hand: {}'.format(4.))
        plt.plot(grid, parameters, color='blue')
        plt.title('Piecewise linear function of integral {}'.format(area))
        plt.savefig('tools/pseudo_visualization/area_under_piecewise_linear_function.png')
        plt.clf()
    
    def test_cast_float_to_uint8(self):
        """Tests the function `cast_float_to_uint8`.
        
        The test is successful if, after the
        data-type cast, the array elements
        belong to [|0, 255|].
        
        """
        array_float = numpy.array([[0.001, -0.001, -2.], [0., 212.651, 255.786]])
        array_uint8 = tls.cast_float_to_uint8(array_float)
        print('Array data-type before the data-type cast: {}'.format(array_float.dtype))
        print('Array elements before the data-type cast:')
        print(array_float)
        print('Array data-type after the data-type cast: {}'.format(array_uint8.dtype))
        print('Array elements after the data-type cast:')
        print(array_uint8)
    
    def test_count_symbols(self):
        """Tests the function `count_symbols`.
        
        The test is successful if, for each set
        of quantized samples, the count of the number
        of occurrences of each symbol is correct.
        
        """
        quantized_samples_0 = numpy.array([0.01, 0.05, -0.03, 0.05, -0.1, -0.1, -0.1, -0.08, 0., -0.05])
        bin_width_0 = 0.01
        quantized_samples_1 = numpy.array([-3., 3., 0., 0., 0., -3., 6., -6., -15., 12.], dtype=numpy.float32)
        bin_width_1 = 3.
        
        hist_0 = tls.count_symbols(quantized_samples_0, bin_width_0)
        hist_1 = tls.count_symbols(quantized_samples_1, bin_width_1)
        print('1st set of quantized samples:')
        print(quantized_samples_0)
        print('1st quantization bin width: {}'.format(bin_width_0))
        print('1st count:')
        print(hist_0)
        print('2nd set of quantized samples:')
        print(quantized_samples_1)
        print('2nd quantization bin width: {}'.format(bin_width_1))
        print('2nd count:')
        print(hist_1)

    def test_count_zero_columns(self):
        """Tests the function `count_zero_columns`.
        
        The test is successful if, for each
        2D array, the function returns the
        right number of zero columns.
        
        """
        arra_2d_0 = numpy.array([[0, 2], [0, 0], [0, 0]], dtype=numpy.int32)
        print('1st 2D array:')
        print(arra_2d_0)
        print('1st number of zero columns: {}'.format(tls.count_zero_columns(arra_2d_0)))
        array_2d_1 = numpy.array([[0., -1., -6.5], [-4.5, 0., 0.]])
        print('2nd 2D array:')
        print(array_2d_1)
        print('2nd number of zero columns: {}'.format(tls.count_zero_columns(array_2d_1)))

    def test_differential_entropy(self):
        """Tests the function `differential_entropy`.
        
        The test is successful if the
        theoretical differential entropy
        is almost equal to the differential
        entropy computed by the function.
        
        """
        nb_points_per_interval = 20
        nb_intervals_per_side = 16
        nb_samples = 20000
        
        nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=0.,
                                                        scale=1.)
        
        # The samples are drawn from the standard normal distribution
        # as the probability density function of the standard normal
        # distribution is chosen to create the parameters of the
        # piecewise linear function.
        samples = numpy.random.normal(loc=0.,
                                      scale=1.,
                                      size=nb_samples)
        theoretical_diff_entropy = (numpy.log(numpy.sqrt(2.*numpy.pi)) + 0.5)/numpy.log(2.)
        diff_entropy = tls.differential_entropy(samples,
                                                parameters,
                                                nb_points_per_interval,
                                                nb_intervals_per_side)
        print('Theoretical differential entropy: {} bits.'.format(theoretical_diff_entropy))
        print('Differential entropy computed by the function: {} bits.'.format(diff_entropy))
    
    def test_discrete_entropy(self):
        """Tests the function `discrete_entropy`.
        
        The test is successful if the five
        entropies computed by the function
        are equal to the entropy computed by hand.
        
        """
        quantized_samples_0 = numpy.array([[0., 128., 12., 128., -4.], [-4., -4., 8., 16., -64.]])
        bin_width_0 = 4.
        quantized_samples_1 = numpy.array([0.1, 0.7, 0.8, 0.7, -2.3, -2.3, -2.3, -0.8, 0.2, -0.4])
        bin_width_1 = 0.1
        quantized_samples_2 = numpy.array([0.75, 1.5, 0., 0., 0., -3., -3.75, -3.75, -0.75, -2.25])
        bin_width_2 = 0.75
        quantized_samples_3 = numpy.array([1.2, 3.6, 3.6, 3.6, -4.8, -6.0, 0., -6.0, -7.2, 7.2])
        bin_width_3 = 1.2
        quantized_samples_4 = numpy.array([[0.02, 0., 0.04, 0.06, -0.12], [-0.24, 0.02, 0., 0., -0.08]])
        bin_width_4 = 0.02
        
        disc_entropy_0 = tls.discrete_entropy(quantized_samples_0,
                                              bin_width_0)
        disc_entropy_1 = tls.discrete_entropy(quantized_samples_1,
                                              bin_width_1)
        disc_entropy_2 = tls.discrete_entropy(quantized_samples_2,
                                              bin_width_2)
        disc_entropy_3 = tls.discrete_entropy(quantized_samples_3,
                                              bin_width_3)
        disc_entropy_4 = tls.discrete_entropy(quantized_samples_4,
                                              bin_width_4)
        entropy_expected = -5*0.1*numpy.log2(0.1) \
                           - 0.2*numpy.log2(0.2) \
                           - 0.3*numpy.log2(0.3)
        print('1st set of quantized samples:')
        print(quantized_samples_0)
        print('1st quantization bin width: {}'.format(bin_width_0))
        print('1st entropy computed by the function: {}'.format(disc_entropy_0))
        print('2nd set of quantized samples:')
        print(quantized_samples_1)
        print('2nd quantization bin width: {}'.format(bin_width_1))
        print('2nd entropy computed by the function: {}'.format(disc_entropy_1))
        print('3rd set of quantized samples:')
        print(quantized_samples_2)
        print('3rd quantization bin width: {}'.format(bin_width_2))
        print('3rd entropy computed by the function: {}'.format(disc_entropy_2))
        print('4th set of quantized samples:')
        print(quantized_samples_3)
        print('4th quantization bin width: {}'.format(bin_width_3))
        print('4th entropy computed by the function: {}'.format(disc_entropy_3))
        print('5th set of quantized samples:')
        print(quantized_samples_4)
        print('5th quantization bin width: {}'.format(bin_width_4))
        print('5th entropy computed by the function: {}'.format(disc_entropy_4))
        print('Entropy computed by hand: {}'.format(entropy_expected))

    def test_expand_parameters(self):
        """Tests the function `expand_parameters`.
        
        The test is successful if the expansion
        adds 4 columns to the left side of the
        matrix storing the parameters of the
        piecewise linear function and 4 columns
        to its right side.
        
        """
        nb_points_per_interval = 4
        nb_intervals_per_side = 2
        nb_added_per_side = 1
        low_projection = 1.e-6
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=0.,
                                                        scale=1.)
        expanded_parameters = tls.expand_parameters(parameters,
                                                    low_projection,
                                                    nb_points_per_interval,
                                                    nb_added_per_side)
        print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
        print('Number of unit intervals added to each side of the grid: {}'.format(nb_added_per_side))
        print('Parameters of the piecewise linear function before the expansion:')
        print(parameters)
        print('Parameters of the piecewise linear function after the expansion:')
        print(expanded_parameters)

    def test_float_to_str(self):
        """Tests the function `float_to_str`.
        
        The test is successful if, for each
        float to be converted, "." is replaced
        by "dot" if the float is not a whole
        number and "-" is replaced by "minus".
        
        """
        float_0 = 2.3
        print('1st float to be converted: {}'.format(float_0))
        print('1st string: {}'.format(tls.float_to_str(float_0)))
        float_1 = -0.01
        print('2nd float to be converted: {}'.format(float_1))
        print('2nd string: {}'.format(tls.float_to_str(float_1)))
        float_2 = 3.
        print('3rd float to be converted: {}'.format(float_2))
        print('3rd string: {}'.format(tls.float_to_str(float_2)))
        float_3 = 0.
        print('4th float to be converted: {}'.format(float_3))
        print('4th string: {}'.format(tls.float_to_str(float_3)))
        float_4 = -4.
        print('5th float to be converted: {}'.format(float_4))
        print('5th string: {}'.format(tls.float_to_str(float_4)))
    
    def test_gradient_density_approximation(self):
        """Tests the function `gradient_density_approximation`.
        
        A histogram is saved at
        "tools/pseudo_visualization/gradient_density_approximation.png".
        The test is successful if the histogram
        absolute values are smaller than 1.e-9.
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 4
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=0.,
                                                        scale=1.)
        
        # There is an intentional mismatch between
        # the probability density function for generating
        # the samples and the probability density function
        # for creating the parameters of the piecewise
        # linear function.
        samples = numpy.random.normal(loc=0.,
                                      scale=0.6,
                                      size=200)
        gradients = tls.gradient_density_approximation(samples,
                                                       parameters,
                                                       nb_points_per_interval,
                                                       nb_intervals_per_side)
        offset = 1.e-4
        approx = numpy.zeros(nb_points)
        for i in range(nb_points):
            parameters_pos = parameters.copy()
            parameters_pos[i] += offset
            loss_pos = tls.loss_density_approximation(samples,
                                                      parameters_pos,
                                                      nb_points_per_interval,
                                                      nb_intervals_per_side)
            parameters_neg = parameters.copy()
            parameters_neg[i] -= offset
            loss_neg = tls.loss_density_approximation(samples,
                                                      parameters_neg,
                                                      nb_points_per_interval,
                                                      nb_intervals_per_side)
            approx[i] = 0.5*(loss_pos - loss_neg)/offset
        
        tls.histogram(gradients - approx,
                      'Gradient checking for the opposite mean probability',
                      'tools/pseudo_visualization/gradient_density_approximation.png')

    def test_gradient_entropy(self):
        """Tests the function `gradient_entropy`.
        
        A histogram is saved at
        "tools/pseudo_visualization/gradient_entropy.png".
        The test is successful if the histogram
        absolute values are smaller than 1.e-9.
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 10
        height_samples = 24
        width_samples = 32
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=0.,
                                                        scale=1.)
        samples = numpy.random.normal(loc=0.,
                                      scale=1.,
                                      size=(height_samples, width_samples))
        gradients = tls.gradient_entropy(samples,
                                         parameters,
                                         nb_points_per_interval,
                                         nb_intervals_per_side)
        
        # `idx_initial` stores the linear piece
        # index of each sample before the gradient
        # checking.
        idx_initial = tls.index_linear_piece(samples.flatten(),
                                             nb_points_per_interval,
                                             nb_intervals_per_side)
        offset = 1.e-4
        approx = numpy.zeros((height_samples, width_samples))
        
        # `is_non_diff_fct` becomes true if the
        # non-differentiability of the piecewise
        # linear function at the edges of pieces
        # wrecks the gradient checking.
        is_non_diff_fct = False
        for i in range(height_samples):
            for j in range(width_samples):
                samples_pos = samples.copy()
                samples_pos[i, j] += offset
                
                # `idx_pos` stores the linear piece
                # index of each sample after adding
                # an offset.
                idx_pos = tls.index_linear_piece(samples_pos.flatten(),
                                                 nb_points_per_interval,
                                                 nb_intervals_per_side)
                diff_entropy_pos = tls.differential_entropy(samples_pos.flatten(),
                                                            parameters,
                                                            nb_points_per_interval,
                                                            nb_intervals_per_side)
                samples_neg = samples.copy()
                samples_neg[i, j] -= offset
                
                # `idx_neg` stores the linear piece
                # index of each sample after subtracting
                # an offset.
                idx_neg = tls.index_linear_piece(samples_neg.flatten(),
                                                 nb_points_per_interval,
                                                 nb_intervals_per_side)
                diff_entropy_neg = tls.differential_entropy(samples_neg.flatten(),
                                                            parameters,
                                                            nb_points_per_interval,
                                                            nb_intervals_per_side)
                approx[i, j] = 0.5*(diff_entropy_pos - diff_entropy_neg)/offset
                is_idx_pos_changed = not numpy.array_equal(idx_initial, idx_pos)
                is_idx_neg_changed = not numpy.array_equal(idx_initial, idx_neg)
                if is_idx_pos_changed or is_idx_neg_changed:
                    is_non_diff_fct = True
        diff = (gradients/height_samples) - approx
        
        if is_non_diff_fct:
            warnings.warn('The non-differentiability of the piecewise linear function wrecks the gradient checking. Re-run it.')
        else:
            tls.histogram(diff.flatten(),
                          'Gradient checking for the differential entropy',
                          'tools/pseudo_visualization/gradient_entropy.png')

    def test_histogram(self):
        """Tests the function `histogram`.
        
        A histogram is saved at
        "tools/pseudo_visualization/histogram.png".
        The test is successful if the selected
        number of bins (60) gives a good histogram
        of 2000 data points.
        
        """
        data = numpy.random.normal(loc=0.,
                                   scale=1.,
                                   size=2000)
        tls.histogram(data,
                      'Standard normal distribution',
                      'tools/pseudo_visualization/histogram.png')

    def test_images_to_rows(self):
        """Tests the function `images_to_rows`.
        
        The test is successful if, for i = 0 ... 3,
        "tools/pseudo_visualization/image_to_rows/images_to_rows_i.png"
        is identical to
        "tools/pseudo_visualization/rows_to_images/rows_to_images_i.png".
        
        """
        images_uint8 = numpy.load('tools/pseudo_data/images_uint8.npy')
        for i in range(images_uint8.shape[3]):
            scipy.misc.imsave('tools/pseudo_visualization/images_to_rows/images_to_rows_{}.png'.format(i),
                              images_uint8[:, :, :, i])
        rows_uint8 = tls.images_to_rows(images_uint8)
        numpy.save('tools/pseudo_data/rows_uint8.npy', rows_uint8)
    
    def test_index_linear_piece(self):
        """Tests the function `index_linear_piece`.
        
        The test is successful if, for each sample,
        the linear piece index of the sample computed
        by the function is equal to the linear piece
        index computed by hand.
        
        """
        nb_points_per_interval = 4
        nb_intervals_per_side = 3
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        
        # The index of the last point of the
        # grid is the linear piece index of
        # samples that are out of the grid
        # bounds. This index does not appear as
        # any "out" sample raises an exception.
        samples = numpy.array([-3., -0.01, 0., 0.01, -1.13, 2.99])
        print('Number of sampling points per unit interval in the grid: {}'.format(nb_points_per_interval))
        print('Number of unit intervals in the right half of the grid: {}'.format(nb_intervals_per_side))
        print('Grid:')
        print(grid)
        print('Samples:')
        print(samples)
        idx_linear_piece = tls.index_linear_piece(samples,
                                                  nb_points_per_interval,
                                                  nb_intervals_per_side)
        print('Linear piece index of each sample computed by the function:')
        print(idx_linear_piece)
        print('Linear piece index of each sample computed by hand:')
        print(numpy.array([0, 11, 12, 12, 7, 23], dtype=numpy.int64))

    def test_kl_divergence(self):
        """Tests the function `kl_divergence`.
        
        A curve is saved at
        "tools/pseudo_visualization/kl_divergence.png".
        The test is successful if the curve
        is convex and its minimum is 0.
        
        """
        nb_points = 201
        z_log_std_squared = numpy.linspace(-5., 5., num=nb_points)
        kl_divergence = numpy.reshape(0.5*(-1. - z_log_std_squared +
            numpy.exp(z_log_std_squared)), (1, nb_points))
        tls.plot_graphs(z_log_std_squared,
                        kl_divergence,
                        'log of the std squared',
                        'KL divergence',
                        ['zero mean'],
                        ['b'],
                        'Evolution of the KL divergence with the log of the std squared',
                        'tools/pseudo_visualization/kl_divergence.png')

    def test_leaky_relu(self):
        """Tests the function `leaky_relu`.
        
        A plot is saved at
        "tools/pseudo_visualization/leaky_relu.png".
        The test is successful if the curve
        of Leaky ReLU is consistent with the
        curve of its derivative.
        
        """
        nb_x = 1001
        x_values = numpy.linspace(-5., 5., num=nb_x)
        y_values = numpy.zeros((2, nb_x))
        y_values[0, :] = tls.leaky_relu(x_values)
        y_values[1, :] = tls.leaky_relu_derivative(x_values)
        tls.plot_graphs(x_values,
                        y_values,
                        '$x$',
                        '$y$',
                        ['$f(x) = $LeakyRelu$(x)$', r'$\partial f(x) / \partial x$'],
                        ['r', 'b'],
                        'LeakyReLU and its derivative',
                        'tools/pseudo_visualization/leaky_relu.png')
    
    def test_loss_density_approximation(self):
        """Tests the function `loss_density_approximation`.
        
        The test is successful if the loss computed by
        the function is close to the loss computed by hand.
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 3
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        parameters = scipy.stats.uniform.pdf(grid,
                                             loc=-1.,
                                             scale=2.)
        samples = numpy.random.uniform(low=-1.,
                                       high=1.,
                                       size=5000)
        
        # The loss below is the sum of two terms. The 1st term
        # must be equal to -1.0. The 2nd term (the integral of
        # the square of the p.d.f of the continuous uniform distribution
        # of support [-1.0, 1.0]) must be equal to 0.5.
        loss_density_approx = tls.loss_density_approximation(samples,
                                                             parameters,
                                                             nb_points_per_interval,
                                                             nb_intervals_per_side)
        print('Loss computed by the function: {}'.format(loss_density_approx))
        print('Loss computed by hand: {}'.format(-0.5))
    
    def test_loss_entropy_reconstruction(self):
        """Tests the function `loss_entropy_reconstruction`.
        
        The test is successful if the entropy-reconstruction
        loss computed by the function is almost equal
        to its approximation.
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 20
        height_visible_units = 4
        width_visible_units = 5
        height_y_tilde = 100
        width_y_tilde = 300
        bin_width = 0.8
        low_projection = 1.e-6
        nb_epochs_fitting = 120
        
        nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        
        # `visible_units` is equal to `reconstruction`.
        # This way, the error between the visible units
        # and their reconstruction is 0.
        visible_units = numpy.ones((height_visible_units, width_visible_units))
        reconstruction = numpy.ones((height_visible_units, width_visible_units))
        y = numpy.random.normal(loc=-1.8,
                                scale=1.,
                                size=(height_y_tilde, width_y_tilde))
        samples_uniform = numpy.random.uniform(low=-0.5*bin_width,
                                               high=0.5*bin_width,
                                               size=(height_y_tilde, width_y_tilde))
        y_tilde = y + samples_uniform
        parameters = fit_piecewise_linear_function(y_tilde.flatten(),
                                                   grid,
                                                   low_projection,
                                                   nb_points_per_interval,
                                                   nb_intervals_per_side,
                                                   nb_epochs_fitting)
        loss = tls.loss_entropy_reconstruction(visible_units,
                                               y_tilde,
                                               reconstruction,
                                               parameters,
                                               nb_points_per_interval,
                                               nb_intervals_per_side,
                                               bin_width,
                                               1.)
        quantized_y = tls.quantization(y, bin_width)
        disc_entropy = tls.discrete_entropy(quantized_y, bin_width)
        print('Entropy-reconstruction loss computed by the function: {}'.format(loss))
        print('Approximation of the entropy-reconstruction loss: {}'.format(disc_entropy))

    def test_mean_psnr(self):
        """Tests the function `mean_psnr`.
        
        The test is successful if the mean
        PSNR computed by the function is almost
        equal to the mean PSNR computed by hand.
        
        """
        nb_examples = 3
        nb_visible = 4
        
        reference_uint8 = 12*numpy.ones((nb_examples, nb_visible), dtype=numpy.uint8)
        reconstruction_uint8 = 15*numpy.ones((nb_examples, nb_visible), dtype=numpy.uint8)
        reconstruction_uint8[2, :] = numpy.array([16, 17, 5, 33], dtype=numpy.uint8)
        psnr = tls.mean_psnr(reference_uint8, reconstruction_uint8)
        print('Mean PNSR computed by the function: {}'.format(psnr))
        print('Mean PSNR computed by hand: {}'.format(34.692405113239))
    
    def test_noise(self):
        """Tests the function `noise`.
        
        A histogram is saved at
        "tools/pseudo_visualization/noise.png".
        The test is successful if the histogram
        looks like that of the uniform distribution
        of support [-0.5, 0.5].
        
        """
        samples = tls.noise(100, 200)
        tls.histogram(samples.flatten(),
                      'Noise from the uniform distribution of support [-0.5, 0.5]',
                      'tools/pseudo_visualization/noise.png')
    
    def test_normed_histogram(self):
        """Tests the function `normed_histogram`.
        
        A normed histogram is saved at
        "tools/pseudo_visualization/normed_histogram.png".
        The test is successful if the drawn
        probability density function (red) fits
        the normed histogram of the data (blue).
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 10
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        pdf = scipy.stats.distributions.norm.pdf(grid,
                                                 loc=0.,
                                                 scale=1.)
        data = numpy.random.normal(loc=0., scale=1., size=5000)
        tls.normed_histogram(data,
                             grid,
                             pdf,
                             'Standard normal distribution',
                             'tools/pseudo_visualization/normed_histogram.png')

    def test_opposite_vlb(self):
        """Tests the function `opposite_vlb`.
        
        The test is successful if the
        opposite of Kingma's approximation
        of the variational lower bound computed
        by the function is almost equal to the
        one computed by hand.
        
        """
        z_mean = numpy.zeros((4, 3))
        z_log_std_squared = numpy.zeros((4, 3))
        height_visible_units = 3
        width_visible_units = 5
        visible_units = numpy.ones((height_visible_units, width_visible_units))
        reconstruction = 0.5*numpy.ones((height_visible_units, width_visible_units))
        alpha = 1.
        print('The KL divergence of the approximate posterior from the prior is equal to 0.')
        
        # 1st case: each visible unit is modeled
        # as a continuous random variable with
        # Gaussian probability density function.
        print('\nVisible unit: continuous random variable with Gaussian probability density function.')
        vlb = tls.opposite_vlb(visible_units,
                               z_mean,
                               z_log_std_squared,
                               reconstruction,
                               alpha,
                               True)
        print('Opposite of Kingma\'s approximation of the vlb computed by the function: {}'.format(vlb))
        print('Opposite of Kingma\'s approximation of the vlb computed by hand: {}'.format(0.625))
        
        # 2nd case: each visible unit is modeled
        # as the probability of activation of a
        # Bernoulli random variable.
        print('\nVisible unit: probability of activation of a Bernoulli random variable.')
        vlb = tls.opposite_vlb(visible_units,
                               z_mean,
                               z_log_std_squared,
                               reconstruction,
                               alpha,
                               False)
        print('Opposite of Kingma\'s approximation of the vlb computed by the function: {}'.format(vlb))
        print('Opposite of Kingma\'s approximation of the vlb computed by hand: {}'.format(3.465736))
    
    def test_plot_graphs(self):
        """Tests the function `plot_graphs`.
        
        A plot is saved at
        "tools/pseudo_visualization/plot_graphs.png".
        The test is successful is the two
        graphs in the plot are consistent
        with the legend.
        
        """
        x_values = numpy.linspace(-5., 5., num=101)
        y_values = numpy.zeros((2, 101))
        y_values[0, :] = 1.7159*numpy.tanh((2./3)*x_values)
        y_values[1, :] = 1./(1. + numpy.exp(-x_values))
        tls.plot_graphs(x_values,
                        y_values,
                        'input',
                        'neural activation',
                        ['scaled tanh', 'sigmoid'],
                        ['b', 'r'],
                        'Evolution of the neural activation with the input',
                        'tools/pseudo_visualization/plot_graphs.png')

    def test_quantization(self):
        """Tests the function `quantization`.
        
        The test is successful if, for each
        set of samples, the quantized samples
        are consistent with the quantization
        bin width.
        
        """
        samples_0 = numpy.array([1.131, 0.112, -2.302], dtype=numpy.float16)
        bin_width_0 = 0.25
        samples_1 = numpy.array([4.1, -2., 9.8], dtype=numpy.float32)
        bin_width_1 = 3.
        samples_2 = numpy.array([1.1, 0., -0.03])
        bin_width_2 = 1.
        
        quantized_samples_0 = tls.quantization(samples_0, bin_width_0)
        quantized_samples_1 = tls.quantization(samples_1, bin_width_1)
        quantized_samples_2 = tls.quantization(samples_2, bin_width_2)
        print('1st set of samples:')
        print(samples_0)
        print('1st quantization bin width: {}'.format(bin_width_0))
        print('1st set of quantized samples:')
        print(quantized_samples_0)
        print('2nd set of samples:')
        print(samples_1)
        print('2nd quantization bin width: {}'.format(bin_width_1))
        print('2nd set of quantized samples:')
        print(quantized_samples_1)
        print('3rd set of samples:')
        print(samples_2)
        print('3rd quantization bin width: {}'.format(bin_width_2))
        print('3rd set of quantized samples:')
        print(quantized_samples_2)

    def test_reconstruction_error(self):
        """Tests the function `reconstruction_error`.
        
        A curve is saved at
        "tools/pseudo_visualization/reconstruction_error.png".
        The test is successful if the curve
        looks like the evolution of the cross
        entropy between a visible unit fixed to
        1.0 and a unit moving from 0.0 to 1.0.
        
        """
        nb_points = 999
        
        # The visible unit is modeled as
        # the probability that a Bernoulli
        # random variable turns on.
        visible_unit = 1.
        reconstruction = numpy.linspace(0.001, 0.999, num=nb_points)
        rec_error = numpy.reshape(-visible_unit*numpy.log(reconstruction) -
            (1. - visible_unit)*numpy.log(1. - reconstruction), (1, nb_points))
        tls.plot_graphs(reconstruction,
                        rec_error,
                        'reconstruction',
                        'reconstruction error',
                        ['visible unit = 1.'],
                        ['b'],
                        'Evolution of the reconstruction error with the reconstruction',
                        'tools/pseudo_visualization/reconstruction_error.png')

    def test_relu(self):
        """Tests the function `relu`.
        
        A plot is saved at
        "tools/pseudo_visualization/relu.png".
        The test is successful if the curve
        of ReLU is consistent with the curve
        of its derivative.
        
        """
        nb_x = 1001
        x_values = numpy.linspace(-5., 5., num=nb_x)
        y_values = numpy.zeros((2, nb_x))
        y_values[0, :] = tls.relu(x_values)
        y_values[1, :] = tls.relu_derivative(x_values)
        tls.plot_graphs(x_values,
                        y_values,
                        '$x$',
                        '$y$',
                        ['$f(x) = $ReLU$(x)$', r'$\partial f(x) / \partial x$'],
                        ['r', 'b'],
                        'ReLU and its derivative',
                        'tools/pseudo_visualization/relu.png')

    def test_rows_to_images(self):
        """Tests the function `rows_to_images`.
        
        The test is successful if, for i = 0 ... 3,
        "tools/pseudo_visualization/rows_to_images/rows_to_images_i.png"
        is identical to
        "tools/pseudo_visualization/images_to_rows/images_to_rows_i.png".
        
        """
        rows_uint8 = numpy.load('tools/pseudo_data/rows_uint8.npy')
        images_uint8 = tls.rows_to_images(rows_uint8, 64, 64)
        for i in range(images_uint8.shape[3]):
            scipy.misc.imsave('tools/pseudo_visualization/rows_to_images/rows_to_images_{}.png'.format(i),
                              images_uint8[:, :, :, i])

    def test_sigmoid(self):
        """Tests the function `sigmoid`.
        
        The test is successful if the activation
        with a protection against numerical overflow
        is identical to the activation without it.
        
        """
        input_0 = numpy.array([[1.21, -9.12], [-0.01, 0.45]])
        input_1 = numpy.array([[-801., -90.], [201., -1.]])
        
        activation_0 = tls.sigmoid(input_0)
        activation_1 = tls.sigmoid(input_1)
        print('1st input to sigmoid:')
        print(input_0)
        print('1st activation with a protection against numerical overflow:')
        print(activation_0)
        print('1st activation without any protection against numerical overflow:')
        print(1./(1. + numpy.exp(-input_0)))
        print('\n2nd input to sigmoid:')
        print(input_1)
        print('2nd activation with a protection against numerical overflow:')
        print(activation_1)
        print('2nd activation without any protection against numerical overflow:')
        print(1./(1. + numpy.exp(-input_1)))
    
    def test_subdivide_set(self):
        """Tests the function `subdivide_set`.
        
        The test is successful if an assertion
        error is raised when the number of
        examples cannot be divided into a
        whole number of mini-batches.
        
        """
        nb_examples = 400
        batch_size = 20
        
        nb_batches = tls.subdivide_set(nb_examples, batch_size)
        print('Number of examples: {}'.format(nb_examples))
        print('Size of the mini-batches: {}'.format(batch_size))
        print('Number of mini-batches: {}'.format(nb_batches))

    def test_visualize_dead(self):
        """Tests the function `visualize_dead`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_dead.png".
        The test is successful if the first
        20 rows of the image are blue, the next
        20 rows are red and the remaining rows
        are black.
        
        """
        quantized_samples = numpy.zeros((200, 200))
        quantized_samples[0:20, :] = -3.5
        quantized_samples[20:40, :] = 0.5
        tls.visualize_dead(quantized_samples,
                           'tools/pseudo_visualization/visualize_dead.png')
    
    def test_visualize_images(self):
        """Tests the function `visualize_images`.
        
        The test is successful if the images
        "tools/pseudo_visualization/images_to_rows_i.png",
        i = 0 ... 3, are arranged in a 4x4 grid,
        providing the image
        "tools/pseudo_visualization/visualize_images.png".
        
        """
        images_uint8 = numpy.load('tools/pseudo_data/images_uint8.npy')
        tls.visualize_images(images_uint8,
                             2,
                             'tools/pseudo_visualization/visualize_images.png')
    
    def test_visualize_rows(self):
        """Tests the function `visualize_rows`.
        
        The test is successful if the image at
        "tools/pseudo_visualization/visualize_images.png"
        is identical to the image at
        "tools/pseudo_visualization/visualize_rows.png".
        
        """
        rows_uint8 = numpy.load('tools/pseudo_data/rows_uint8.npy')
        tls.visualize_rows(rows_uint8,
                           64,
                           64,
                           2,
                           'tools/pseudo_visualization/visualize_rows.png')

    def test_visualize_weights(self):
        """Tests the function `visualize_weights`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_weights.png".
        The test is successful if the top squares
        in the image are yellow and the bottom
        squares are blue.
        
        """
        height_image = 32
        width_image = 32
        nb_pixels_per_channel = height_image*width_image
        yellow_triplet = numpy.concatenate(
            (0.8*numpy.ones((3, nb_pixels_per_channel)),
            0.8*numpy.ones((3, nb_pixels_per_channel)),
            -0.2*numpy.ones((3, nb_pixels_per_channel))), axis=1)
        blue_triplet = numpy.concatenate(
            (-0.2*numpy.ones((3, nb_pixels_per_channel)),
            -0.2*numpy.ones((3, nb_pixels_per_channel)),
            0.8*numpy.ones((3, nb_pixels_per_channel))), axis=1)
        weights = numpy.concatenate((yellow_triplet, blue_triplet), axis=0)
        tls.visualize_weights(weights,
                              height_image,
                              width_image,
                              2,
                              'tools/pseudo_visualization/visualize_weights.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library that contains common tools.')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterTools()
    getattr(tester, 'test_' + args.name)()


