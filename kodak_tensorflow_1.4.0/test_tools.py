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

import tools.tools as tls


class TesterTools(object):
    """Class for testing the library that contains common tools."""
    
    def test_average_entropies(self):
        """Tests the function `average_entropies`.
        
        The test is successful if the mean differential
        entropy of the data is close to the mean entropy
        of the quantized data.
        
        """
        bin_widths = numpy.array([1., 1.], dtype=numpy.float32)
        loc_norm_0 = -3.
        scale_norm_0 = 4.
        loc_norm_1 = 2.
        scale_norm_1 = 1.
        
        data_0 = numpy.random.normal(loc=loc_norm_0,
                                     scale=scale_norm_0,
                                     size=(2, 12, 20, 1)).astype(numpy.float32)
        data_1 = numpy.random.normal(loc=loc_norm_1,
                                     scale=scale_norm_1,
                                     size=(2, 12, 20, 1)).astype(numpy.float32)
        theoretical_diff_entropy_0 = (0.5 + numpy.log(scale_norm_0*numpy.sqrt(2.*numpy.pi)))/numpy.log(2.)
        theoretical_diff_entropy_1 = (0.5 + numpy.log(scale_norm_1*numpy.sqrt(2.*numpy.pi)))/numpy.log(2.)
        mean_diff_entropy = 0.5*(theoretical_diff_entropy_0 + theoretical_diff_entropy_1)
        data = numpy.concatenate((data_0, data_1), axis=3)
        avg_entropies = tls.average_entropies(data, bin_widths)
        
        # When the quantization bin widths are equal
        # to 1.0, the differential entropy of the
        # data should be close to the entropy of
        # the quantized data.
        print('Quantization bin widths:')
        print(bin_widths)
        print('Mean differential entropy of the data: {}'.format(mean_diff_entropy))
        print('Mean entropy of the quantized data: {}'.format(avg_entropies))
    
    def test_cast_bt601(self):
        """Tests the function `cast_bt601`.
        
        The test is successful if, after the
        data-type cast, the array elements
        belong to [|16, 235|].
        
        """
        array_float = numpy.array([[15.431, -0.001, 0.], [235.678, 143.18, 1.111]], dtype=numpy.float32)
        array_uint8 = tls.cast_bt601(array_float)
        print('Array data-type before the data-type cast: {}'.format(array_float.dtype))
        print('Array elements before the data-type cast:')
        print(array_float)
        print('Array data-type after the data-type cast: {}'.format(array_uint8.dtype))
        print('Array elements after the data-type cast:')
        print(array_uint8)
    
    def test_cast_float_to_int16(self):
        """Tests the function `cast_float_to_int16`.
        
        The test is successful if an exception
        is raised when an array element before
        the data-type cast is strictly larger than
        32767.5 in absolute value.
        
        """
        array_float = numpy.array([[32767.49, -32766.87, 0.], [-1.1, -2.2, 0.001]], dtype=numpy.float32)
        array_int16 = tls.cast_float_to_int16(array_float)
        print('Array data-type before the data-type cast: {}'.format(array_float.dtype))
        print('Array elements before the data-type cast:')
        print(array_float)
        print('Array data-type after the data-type cast: {}'.format(array_int16.dtype))
        print('Array elements after the data-type cast:')
        print(array_int16)
    
    def test_convert_approx_entropy(self):
        """Tests the function `convert_approx_entropy`.
        
        The test is successful if the mean
        approximate entropy is 1.0.
        
        """
        scaled_approx_entropy = numpy.float64(128.)
        gamma_scaling = 64.
        nb_maps = 2
        
        mean_approx_entropy = tls.convert_approx_entropy(scaled_approx_entropy,
                                                         gamma_scaling,
                                                         nb_maps)
        print('Scaled approximate entropy: {}'.format(scaled_approx_entropy))
        print('Scaling coefficient: {}'.format(gamma_scaling))
        print('Number of latent variable feature maps: {}'.format(nb_maps))
        print('Mean approximate entropy: {}'.format(mean_approx_entropy))
    
    def test_count_symbols(self):
        """Tests the function `count_symbols`.
        
        The test is successful if, for each set of
        quantized samples, the count of the number
        of occurrences of each symbol is correct.
        
        """
        quantized_samples_0 = numpy.array([0.01, 0.05, -0.03, 0.05, -0.1, -0.1, -0.1, -0.08, 0., -0.05])
        bin_width_0 = 0.01
        quantized_samples_1 = numpy.array([-3., 3., 0., 0., 0., -3., 6., -6., -15., 12.])
        bin_width_1 = 3.
        samples_2 = numpy.random.normal(loc=0., scale=1., size=6).astype(numpy.float32)
        bin_width_2 = 1.2938486
        quantized_samples_2 = bin_width_2*numpy.round(samples_2/bin_width_2)
        
        hist_0 = tls.count_symbols(quantized_samples_0, bin_width_0)
        hist_1 = tls.count_symbols(quantized_samples_1, bin_width_1)
        hist_2 = tls.count_symbols(quantized_samples_2, bin_width_2)
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
        print('3rd set of quantized samples:')
        print(quantized_samples_2)
        print('3rd quantization bin width: {}'.format(bin_width_2))
        print('3rd count:')
        print(hist_2)
    
    def test_crop_option_2d(self):
        """Tests the function `crop_option_2d`.
        
        A 1st image is saved at
        "tools/pseudo_visualization/crop_option_2d/luminance_image.png".
        A 2nd image is saved at
        "tools/pseudo_visualization/crop_option_2d/crop_random.png".
        A 3rd image is saved at
        "tools/pseudo_visualization/crop_option_2d/crop_center.png".
        The test is successful if the 2nd image
        is a random crop of the 1st image and the
        3rd image is a central crop of the 1st image.
        
        """
        width_crop = 64
        
        luminance_uint8 = numpy.load('tools/pseudo_data/luminances_uint8.npy')[0, :, :, 0]
        crop_0 = tls.crop_option_2d(luminance_uint8,
                                    width_crop,
                                    True)
        crop_1 = tls.crop_option_2d(luminance_uint8,
                                    width_crop,
                                    False)
        scipy.misc.imsave('tools/pseudo_visualization/crop_option_2d/luminance_image.png',
                          luminance_uint8)
        scipy.misc.imsave('tools/pseudo_visualization/crop_option_2d/crop_random.png',
                          crop_0)
        scipy.misc.imsave('tools/pseudo_visualization/crop_option_2d/crop_center.png',
                          crop_1)
    
    def test_crop_repeat_2d(self):
        """Tests the function `crop_repeat_2d`.
        
        A 1st image is saved at
        "tools/pseudo_visualization/crop_repeat_2d/luminance_image.png".
        A 2nd image is saved at
        "tools/pseudo_visualization/crop_repeat_2d/crop_0.png".
        A 3rd image is saved at
        "tools/pseudo_visualization/crop_repeat_2d/crop_1.png".
        The test is successful if the 2nd image
        is an enlarged crop of the bottom-left
        of the 1st image and the 3rd image is
        an enlarged crop of the top-right of
        the 1st image.
        
        """
        image_uint8 = numpy.load('tools/pseudo_data/luminances_uint8.npy')[0, :, :, 0]
        crop_0 = tls.crop_repeat_2d(image_uint8, image_uint8.shape[0] - 81, 0)
        crop_1 = tls.crop_repeat_2d(image_uint8, 0, image_uint8.shape[1] - 81)
        scipy.misc.imsave('tools/pseudo_visualization/crop_repeat_2d/luminance_image.png',
                          image_uint8)
        scipy.misc.imsave('tools/pseudo_visualization/crop_repeat_2d/crop_0.png',
                          crop_0)
        scipy.misc.imsave('tools/pseudo_visualization/crop_repeat_2d/crop_1.png',
                          crop_1)
    
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
    
    def test_histogram(self):
        """Tests the function `histogram`.
        
        A histogram is saved at
        "tools/pseudo_visualization/histogram.png".
        The test is successful if the selected
        number of bins (60) gives a good histogram
        of 2000 data points.
        
        """
        data = numpy.random.normal(loc=0., scale=1., size=2000)
        tls.histogram(data,
                      'Standard normal distribution',
                      'tools/pseudo_visualization/histogram.png')
    
    def test_jensen_shannon_divergence(self):
        """Tests the function `jensen_shannon_divergence`.
        
        The test is successful if the Jensen-Shannon
        divergence between Dnorm and Dnorm is 0.0 and
        the Jensen-Shannon divergence between Dnorm and
        Dunif is strictly positive.
        
        """
        # The standard deviation of the normal
        # distribution is small so that no symbol
        # has probability 0.0.
        data = numpy.random.normal(loc=0., scale=0.5, size=300000)
        middle_1st_bin = numpy.round(numpy.amin(data))
        middle_last_bin = numpy.round(numpy.amax(data))
        nb_edges = int(middle_last_bin - middle_1st_bin) + 2
        bin_edges = numpy.linspace(middle_1st_bin - 0.5,
                                   middle_last_bin + 0.5,
                                   num=nb_edges)
        probs_0 = numpy.histogram(data,
                                  bins=bin_edges,
                                  density=True)[0]
        probs_1 = (1./(nb_edges - 1))*numpy.ones(nb_edges - 1)
        divergence_0 = tls.jensen_shannon_divergence(probs_0, probs_0)
        divergence_1 = tls.jensen_shannon_divergence(probs_0, probs_1)
        print('The two distributions below are estimated via sampling.')
        print('Dnorm: normal distribution with mean 0.0 and standard deviation 0.5.')
        print('Dunif: uniform distribution.')
        print('Jensen-Shannon divergence between Dnorm and Dnorm: {}'.format(divergence_0))
        print('Jensen-Shannon divergence between Dnorm and Dunif: {}'.format(divergence_1))
    
    def test_normed_histogram(self):
        """Tests the function `normed_histogram`.
        
        A 1st normed histogram is saved at
        "tools/pseudo_visualization/normed_histogram/normed_histogram_0.png".
        A second normed histogram is saved at
        "tools/pseudo_visualization/normed_histogram/normed_histogram_1.png".
        A 3rd normed histogram is saved at
        "tools/pseudo_visualization/normed_histogram/normed_histogram_2.png".
        The test is successful if, for each
        normed histogram, the drawn probability
        density function (red) fits the normed
        histogram of the data (blue).
        
        """
        nb_points_per_interval = 10
        nb_intervals_per_side = 25
        batch_size = 2
        height = 10
        width = 50
        loc_norm_0 = 0.
        loc_norm_1 = -6.
        loc_norm_2 = 4.
        scale_norm_0 = 1.
        scale_norm_1 = 0.1
        scale_norm_2 = 3.
        titles = [
            'Normal distribution of mean {0} and std {1}'.format(loc_norm_0, scale_norm_0),
            'Normal distribution of mean {0} and std {1}'.format(loc_norm_1, scale_norm_1),
            'Normal distribution of mean {0} and std {1}'.format(loc_norm_2, scale_norm_2),
        ]
        paths = [
            'tools/pseudo_visualization/normed_histogram/normed_histogram_0.png',
            'tools/pseudo_visualization/normed_histogram/normed_histogram_1.png',
            'tools/pseudo_visualization/normed_histogram/normed_histogram_2.png'
        ]
        
        nb_points = 2*nb_intervals_per_side*nb_points_per_interval + 1
        grid = numpy.linspace(-nb_intervals_per_side,
                              nb_intervals_per_side,
                              num=nb_points)
        pdf_norm_0 = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=loc_norm_0,
                                                        scale=scale_norm_0)
        pdf_norm_1 = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=loc_norm_1,
                                                        scale=scale_norm_1)
        pdf_norm_2 = scipy.stats.distributions.norm.pdf(grid,
                                                        loc=loc_norm_2,
                                                        scale=scale_norm_2)
        reshaped_pdf_norm_0 = numpy.reshape(pdf_norm_0, (1, nb_points))
        reshaped_pdf_norm_1 = numpy.reshape(pdf_norm_1, (1, nb_points))
        reshaped_pdf_norm_2 = numpy.reshape(pdf_norm_2, (1, nb_points))
        pdfs = numpy.concatenate((reshaped_pdf_norm_0, reshaped_pdf_norm_1, reshaped_pdf_norm_2),
                                 axis=0)
        data_norm_0 = numpy.random.normal(loc=loc_norm_0,
                                          scale=scale_norm_0,
                                          size=(batch_size, height, width))
        data_norm_1 = numpy.random.normal(loc=loc_norm_1,
                                          scale=scale_norm_1,
                                          size=(batch_size, height, width))
        data_norm_2 = numpy.random.normal(loc=loc_norm_2,
                                          scale=scale_norm_2,
                                          size=(batch_size, height, width))
        
        # `pdfs[i, :]` is the sampled probability
        # density function that is associated to the
        # `data[:, :, :, i]`.
        data = numpy.zeros((batch_size, height, width, 3))
        data[:, :, :, 0] = data_norm_0
        data[:, :, :, 1] = data_norm_1
        data[:, :, :, 2] = data_norm_2
        tls.normed_histogram(data,
                             grid,
                             pdfs,
                             titles,
                             paths)
    
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
    
    def test_psnr_2d(self):
        """Tests the function `psnr_2d`.
        
        The test is successful if the PSNR
        computed by hand is equal to the
        PSNR computed by the function.
        
        """
        height = 2
        width = 2
        
        reference_uint8 = 12*numpy.ones((height, width), dtype=numpy.uint8)
        reconstruction_uint8 = 15*numpy.ones((height, width), dtype=numpy.uint8)
        psnr = tls.psnr_2d(reference_uint8, reconstruction_uint8)
        print('PSNR computed by hand: {}'.format(38.5883785143))
        print('PNSR computed by the function: {}'.format(psnr))
    
    def test_quantize_per_map(self):
        """Tests the function `quantize_per_map`.
        
        The test is successful if, for each map
        of data, the map of quantized data is
        consistent with the quantization bin width.
        
        """
        data = numpy.random.normal(loc=0., scale=5., size=(2, 1, 3, 2)).astype(numpy.float32)
        bin_widths = numpy.array([1., 10.], dtype=numpy.float32)
        quantized_data = tls.quantize_per_map(data, bin_widths)
        print('1st map of data after being flattened:')
        print(data[:, :, :, 0].flatten())
        print('1st quantization bin width: {}'.format(bin_widths[0]))
        print('1st map of quantized data after being flattened:')
        print(quantized_data[:, :, :, 0].flatten())
        print('2nd map of data after being flattened:')
        print(data[:, :, :, 1].flatten())
        print('2nd quantization bin width: {}'.format(bin_widths[1]))
        print('2nd map of quantized data after being flattened:')
        print(quantized_data[:, :, :, 1].flatten())
    
    def test_rate_3d(self):
        """Tests the function `rate_3d`.
        
        The test is successful if the rate
        computed by the function is close to
        the theoretical rate.
        
        """
        h_in = 768
        w_in = 512
        height_map = 256
        width_map = 256
        
        # The quantization bin widths are small
        # so that the comparison between the
        # theoretical rate and the rate computed
        # by the function is precise enough.
        bin_widths = numpy.array([0.2, 0.5], dtype=numpy.float32)
        expanded_latent_float32 = numpy.random.normal(loc=0.,
                                                      scale=1.,
                                                      size=(1, height_map, width_map, 2)).astype(numpy.float32)
        expanded_quantized_latent_float32 = tls.quantize_per_map(expanded_latent_float32, bin_widths)
        quantized_latent_float32 = numpy.squeeze(expanded_quantized_latent_float32, axis=0)
        rate = tls.rate_3d(quantized_latent_float32,
                           bin_widths,
                           h_in,
                           w_in)
        
        # The equation below is derived from the
        # theorem 8.3.1 in the book
        # "Elements of information theory", 2nd edition,
        # written by Thomas M. Cover and Joy A. Thomas.
        theoretical_entropy = -numpy.log2(bin_widths[0]) - numpy.log2(bin_widths[1]) + (numpy.log(2.*numpy.pi) + 1.)/numpy.log(2.)
        theoretical_rate = theoretical_entropy*height_map*width_map/(h_in*w_in)
        print('Rate computed by the function: {}'.format(rate))
        print('Theoretical rate: {}'.format(theoretical_rate))
    
    def test_rgb_to_ycbcr(self):
        """Tests the function `rgb_to_ycbcr`.
        
        The test is successful if, for each
        channel Y, Cb and Cr, the channel computed
        by hand is equal to the channel computed by
        the function.
        
        """
        red_uint8 = numpy.array([[1, 214, 23], [45, 43, 0]], dtype=numpy.uint8)
        green_uint8 = numpy.array([[255, 255, 23], [0, 13, 0]], dtype=numpy.uint8)
        blue_uint8 = numpy.array([[100, 255, 0], [0, 0, 0]], dtype=numpy.uint8)
        rgb_uint8 = numpy.stack((red_uint8, green_uint8, blue_uint8), axis=2)
        ycbcr_uint8 = tls.rgb_to_ycbcr(rgb_uint8)
        print('Red channel:')
        print(red_uint8)
        print('Green channel:')
        print(green_uint8)
        print('Blue channel:')
        print(blue_uint8)
        print('Luminance computed by the function:')
        print(ycbcr_uint8[:, :, 0])
        print('Luminance computed by hand:')
        print(numpy.array([[155, 224, 34], [28, 34, 16]], dtype=numpy.uint8))
        print('Blue chrominance computed by the function:')
        print(ycbcr_uint8[:, :, 1])
        print('Blue chrominance computed by hand:')
        print(numpy.array([[98, 134, 118], [121, 118, 128]], dtype=numpy.uint8))
        print('Red chrominance computed by the function:')
        print(ycbcr_uint8[:, :, 2])
        print('Red chrominance computed by hand:')
        print(numpy.array([[28, 110, 130], [148, 142, 128]], dtype=numpy.uint8))
    
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
    
    def test_tile_cauchy(self):
        """Tests the function `tile_cauchy`.
        
        A plot is saved at
        "tools/pseudo_visualization/tile_cauchy.png".
        The test is successful if the plot
        looks like the Cauchy probability
        density function.
        
        """
        grid = numpy.linspace(-10., 10., num=101, dtype=numpy.float32)
        reps = 2
        
        # `pdfs.dtype` is equal to `numpy.float32`.
        pdfs = tls.tile_cauchy(grid, reps)
        plt.plot(grid, pdfs[0, :], color='blue')
        plt.title('Cauchy probability density function')
        plt.savefig('tools/pseudo_visualization/tile_cauchy.png')
        plt.clf()
    
    def test_untar_archive(self):
        """Tests the function `untar_archive`.
        
        The test is successful if the folder
        "tools/pseudo_visualization/untar_archive/"
        contains "rgb_tree.jpg" and "rgb_artificial.png".
        
        """
        path_to_root = 'tools/pseudo_visualization/untar_archive/'
        path_to_tar = 'tools/pseudo_data/pseudo_archive.tar'
        
        tls.untar_archive(path_to_root, path_to_tar)
    
    def test_visualize_crops(self):
        """Tests the function `visualize_crops`.
        
        A 1st image is saved at
        "tools/pseudo_visualization/visualize_crops/crop_0.png".
        A 2nd image is saved at
        "tools/pseudo_visualization/visualize_crops/crop_1.png".
        The test is successful if the 1st
        image is identical to the image at
        "tools/pseudo_visualization/crop_repeat_2d/crop_0.png"
        and the 2nd image is identical to
        the image at
        "tools/pseudo_visualization/crop_repeat_2d/crop_1.png".
        
        """
        paths = [
            'tools/pseudo_visualization/visualize_crops/crop_0.png',
            'tools/pseudo_visualization/visualize_crops/crop_1.png'
        ]
        
        image_uint8 = numpy.load('tools/pseudo_data/luminances_uint8.npy')[0, :, :, 0]
        positions_top_left = numpy.array([[image_uint8.shape[0] - 81, 0], [0, image_uint8.shape[1] - 81]],
                                         dtype=numpy.int32)
        tls.visualize_crops(image_uint8,
                            positions_top_left,
                            paths)
    
    def test_visualize_luminances(self):
        """Tests the function `visualize_luminances`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_luminances.png".
        The test is successful if this
        image contains 4 luminance images
        which are arranged in a 4x4 grid.
        
        """
        luminances_uint8 = numpy.load('tools/pseudo_data/luminances_uint8.npy')
        tls.visualize_luminances(luminances_uint8,
                                 2,
                                 'tools/pseudo_visualization/visualize_luminances.png')
    
    def test_visualize_representation(self):
        """Tests the function `visualize_representation`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_representation.png".
        The test is successful if the image
        contains 4 squares. The top-right square
        is black and the top-left square is white.
        The other two squares are grey.
        
        """
        height_r = 80
        width_r = 80
        
        representation_float32 = numpy.zeros((height_r, width_r, 4), dtype=numpy.float32)
        representation_float32[:, :, 0] = -46.*numpy.ones((height_r, width_r), dtype=numpy.float32)
        representation_float32[:, :, 1] = 240.*numpy.ones((height_r, width_r), dtype=numpy.float32)
        representation_float32[:, :, 2] = 90.*numpy.ones((height_r, width_r), dtype=numpy.float32)
        representation_float32[:, :, 3] = numpy.zeros((height_r, width_r), dtype=numpy.float32)
        tls.visualize_representation(representation_float32,
                                     2,
                                     'tools/pseudo_visualization/visualize_representation.png')
    
    def test_visualize_rotated_luminance(self):
        """Tests the function `visualize_rotated_luminance`.
        
        A 1st image is saved at
        "tools/pseudo_visualization/visualize_rotated_luminance/luminance_rotation.png".
        A 2nd image is saved at
        "tools/pseudo_visualization/visualize_rotated_luminance/luminance_crop_0.png".
        A 3rd image is saved at
        "tools/pseudo_visualization/visualize_rotated_luminance/luminance_crop_1.png".
        The test is successful if the 2nd and the 3rd
        image are two distinct crops of the 1st image.
        
        """
        positions_top_left = numpy.array([[10, 10], [200, 200]], dtype=numpy.int32)
        paths = [
            'tools/pseudo_visualization/visualize_rotated_luminance/luminance_rotation.png',
            'tools/pseudo_visualization/visualize_rotated_luminance/luminance_crop_0.png',
            'tools/pseudo_visualization/visualize_rotated_luminance/luminance_crop_1.png'
        ]
        
        rgb_uint8 = scipy.misc.imread('tools/pseudo_data/rgb_web.jpg')
        luminance_before_rotation_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, 0]
        tls.visualize_rotated_luminance(luminance_before_rotation_uint8,
                                        True,
                                        positions_top_left,
                                        paths)
    
    def test_visualize_weights(self):
        """Tests the function `visualize_weights`.
        
        An image is saved at
        "tools/pseudo_visualization/visualize_weights.png".
        The test is successful if the image
        contains 4 squares. The top-right square
        is black and the top-left square is white.
        The other two squares are grey.
        
        """
        height_w = 80
        width_w = 80
        
        weights = numpy.zeros((height_w, width_w, 1, 4), dtype=numpy.float32)
        weights[:, :, 0, 0] = -46.*numpy.ones((height_w, width_w), dtype=numpy.float32)
        weights[:, :, 0, 1] = 240.*numpy.ones((height_w, width_w), dtype=numpy.float32)
        weights[:, :, 0, 2] = 10.*numpy.ones((height_w, width_w), dtype=numpy.float32)
        weights[:, :, 0, 3] = 160.*numpy.ones((height_w, width_w), dtype=numpy.float32)
        tls.visualize_weights(weights,
                              2,
                              'tools/pseudo_visualization/visualize_weights.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library that contains common tools.')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    tester = TesterTools()
    getattr(tester, 'test_' + args.name)()


