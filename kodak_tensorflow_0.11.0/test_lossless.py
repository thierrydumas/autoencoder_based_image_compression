"""A script to test the libraries in the folder "lossless"."""

import argparse
import numpy
import scipy.misc

import lossless.compression
import lossless.interface_cython
import lossless.stats
import tools.tools as tls


class TesterLossless(object):
    """Class for testing the libraries in the folder "lossless"."""
    
    def test_compress_lossless_maps(self):
        """Tests the function `compress_lossless_maps` in the file "lossless/compression.py".
        
        The test is successful if, for each
        centered-quantized latent variable feature
        map, the coding cost computed by the function
        is larger than the theoretical (minimum)
        coding cost while being relatively close
        to the theoretical coding cost.
        
        """
        height_map = 384
        width_map = 384
        
        # The quantization bin widths are small
        # so that the comparison between the
        # theoretical (minimum) coding cost and
        # the coding cost computed by the function
        # is precise enough.
        bin_widths_test = numpy.array([0.5, 0.25], dtype=numpy.float32)
        laplace_scales = numpy.array([0.5, 3.], dtype=numpy.float32)
        
        # Note that the binary probabilities saved at
        # "lossless/pseudo_data/binary_probabilities_compress_maps_0.npy"
        # and those saved at
        # "lossless/pseudo_data/binary_probabilities_compress_maps_1.npy"
        # are specific to the three Laplace distributions
        # below. This means that the binary probabilities
        # must be modified if `laplace_scales` is modified.
        paths_to_binary_probabilities = [
            'lossless/pseudo_data/binary_probabilities_compress_maps_0.npy',
            'lossless/pseudo_data/binary_probabilities_compress_maps_1.npy'
        ]
        
        centered_data_0 = numpy.random.laplace(loc=0.,
                                               scale=laplace_scales[0].item(),
                                               size=(1, height_map, width_map, 1)).astype(numpy.float32)
        centered_data_1 = numpy.random.laplace(loc=0.,
                                               scale=laplace_scales[1].item(),
                                               size=(1, height_map, width_map, 1)).astype(numpy.float32)
        centered_data = numpy.concatenate((centered_data_0, centered_data_1),
                                          axis=3)
        expanded_centered_quantized_data = tls.quantize_per_map(centered_data, bin_widths_test)
        centered_quantized_data = numpy.squeeze(expanded_centered_quantized_data,
                                                axis=0)
        tiled_bin_widths = numpy.tile(numpy.reshape(bin_widths_test, (1, 1, 2)),
                                      (height_map, width_map, 1))
        ref_int16 = tls.cast_float_to_int16(centered_quantized_data/tiled_bin_widths)
        (rec_int16_0, nb_bits_each_map_0) = \
            lossless.compression.compress_lossless_maps(ref_int16,
                                                        paths_to_binary_probabilities[0])
        numpy.testing.assert_equal(ref_int16,
                                   rec_int16_0,
                                   err_msg='The test fails as the lossless compression alters the signed integers.')
        (rec_int16_1, nb_bits_each_map_1) = \
            lossless.compression.compress_lossless_maps(ref_int16,
                                                        paths_to_binary_probabilities[1])
        numpy.testing.assert_equal(ref_int16,
                                   rec_int16_1,
                                   err_msg='The test fails as the lossless compression alters the signed integers.')
        
        # The equation below is derived from the
        # theorem 8.3.1 in the book
        # "Elements of information theory", 2nd edition,
        # written by Thomas M. Cover and Joy A. Thomas.
        theoretical_entropies = -numpy.log2(bin_widths_test) + (numpy.log(2.*laplace_scales) + 1.)/numpy.log(2.)
        print('B0 denotes the binary probabilities saved at "{}".'.format(paths_to_binary_probabilities[0]))
        print('B1 denotes the binary probabilities saved at "{}".'.format(paths_to_binary_probabilities[1]))
        print('\n1st centered-quantized latent variable feature map.')
        print('Theoretical coding cost: {} bits.'.format(theoretical_entropies[0]*height_map*width_map))
        print('Coding cost computed by the function via B0: {} bits.'.format(nb_bits_each_map_0[0]))
        print('Coding cost computed by the function via B1: {} bits.'.format(nb_bits_each_map_1[0]))
        print('\n2nd centered-quantized latent variable feature map.')
        print('Theoretical coding cost: {} bits.'.format(theoretical_entropies[1]*height_map*width_map))
        print('Coding cost computed by the function via B0: {} bits.'.format(nb_bits_each_map_0[1]))
        print('Coding cost computed by the function via B1: {} bits.'.format(nb_bits_each_map_1[1]))
    
    def test_compress_lossless_flattened_map(self):
        """Tests the function `compress_lossless_flattened_map` in the file "lossless/interface_cython.pyx".
        
        The test is successful if no exception is raised,
        meaning that the compression is lossless.
        
        """
        ref_map_int16 = numpy.array([0, 1, -2, 2, 1, 0, 0, 0], dtype=numpy.int16)
        probabilities = numpy.array([0.5, 0.5, 0.5])
        rec_map_int16 = lossless.interface_cython.compress_lossless_flattened_map(ref_map_int16, probabilities)[0]
        numpy.testing.assert_equal(ref_map_int16,
                                   rec_map_int16,
                                   err_msg='The test fails as the lossless compression alters the signed integers.')
    
    def test_compute_binary_probabilities(self):
        """Tests the function `compute_binary_probabilities` in the file "lossless/stats.py".
        
        The test is successful if, for each set of
        test quantization bin widths, for each absolute
        centered-quantized latent variable feature map,
        the binary probabilities computed by the function
        are close to the binary probabilities computed
        by hand.
        
        """
        nb_images = 100
        height_map = 32
        width_map = 48
        bin_widths_test_0 = numpy.array([2., 2., 2.], dtype=numpy.float32)
        bin_widths_test_1 = numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32)
        truncated_unary_length = 4
        
        y_float32_0 = numpy.random.uniform(low=-10.,
                                           high=10.,
                                           size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_1 = numpy.random.laplace(loc=0.5,
                                           scale=2.5,
                                           size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_2 = numpy.random.standard_cauchy(size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32 = numpy.concatenate((y_float32_0, y_float32_1, y_float32_2),
                                      axis=3)
        map_mean = numpy.array([0., 0.5, 0.], dtype=numpy.float32)
        binary_probabilities_0 = lossless.stats.compute_binary_probabilities(y_float32,
                                                                             bin_widths_test_0,
                                                                             map_mean,
                                                                             truncated_unary_length)
        print('1st set of test quantization bin widths:')
        print(bin_widths_test_0)
        print('1st absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_0[0, :])
        
        # Let x be a continuous random variable following the
        # uniform distribution of support [-10.0, 10.0]. The
        # probability the 1st binary decision is 0 is written
        # p(|x| <= 1.0) = 1.0/10. The probability the 2nd
        # binary decision is 0 is written
        # p(1.0 <= |x| <= 3.0)/p(|x| >= 1.0) = (2.0/10)/(9.0/10) = 2.0/9.
        # The probability the 3rd binary decision is 0 is written
        # p(3.0 <= |x| <= 5.0)/p(|x| >= 3.0) = (2.0/10)/(7.0/10) = 2.0/7.
        # The above calculations use the cumulative distribution
        # function of the uniform distribution of support [-10.0, 10.0].
        print('Binary probabilities computed by hand:')
        print([1./10, 2./9, 2./7, 2./5])
        print('2nd absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_0[1, :])
        
        # Let x be a continuous random variable following the
        # Laplace distribution of mean 0.0 and scale 2.5. It is
        # said `mean 0.0` as the 2nd latent variable feature map
        # is centered before being quantized. The probability
        # the 1st binary decision is 0 is written
        # p(|x| <= 1.0) = 0.3297. The probability the 2nd binary
        # decision is 0 is written
        # p(1.0 <= |x| <= 3.0)/p(|x| >= 1.0) = 0.3691/0.6703 = 0.5507.
        # The probability the 3rd binary decision is 0 is written
        # p(3.0 <= |x| <= 5.0)/p(|x| >= 3.0) = 0.1659/0.3012 = 0.5507.
        # The above calculations use the cumulative distribution
        # function of the Laplace distribution of mean 0 and scale 2.5.
        print('Binary probabilities computed by hand:')
        print([0.3297, 0.5507, 0.5507, 0.5507])
        print('3rd absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_0[2, :])
        
        # Let x be a continuous random variable following the
        # standard Cauchy distribution. The probability the 1st
        # binary decision is 0 is written p(|x| <= 1.0) = 0.5.
        # The probability the 2nd binary decision is 0 is written
        # p(1.0 <= |x| <= 3.0)/p(|x| >= 1.0) = 0.2952/0.5 = 0.5903.
        # The probability the 3rd binary decision is 0 is written
        # p(3.0 <= |x| <= 5.0)/p(|x| >= 3.0) = 0.079/0.2048 = 0.3865.
        # The above calculations use the cumulative distribution
        # function of the standard Cauchy distribution.
        print('Binary probabilities computed by hand:')
        print([0.5, 0.5903, 0.3865, 0.2811])
        
        binary_probabilities_1 = lossless.stats.compute_binary_probabilities(y_float32,
                                                                             bin_widths_test_1,
                                                                             map_mean,
                                                                             truncated_unary_length)
        print('\n2nd set of test quantization bin widths:')
        print(bin_widths_test_1)
        print('1st absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_1[0, :])
        
        # Let x be a continuous random variable following the
        # uniform distribution of support [-10.0, 10.0]. The
        # probability the 1st binary decision is 0 is written
        # p(|x| <= 0.25) = 1.0/40. The probability the 2nd
        # binary decision is 0 is written
        # p(0.25 <= |x| <= 0.75)/p(|x| >= 0.25) = (2.0/40)/(39.0/40) = 2.0/39.
        # The probability the 3rd binary decision is 0 is written
        # p(0.75 <= |x| <= 1.25)/p(|x| >= 0.75) = (2.0/40)/(37.0/40) = 2.0/37.
        print('Binary probabilities computed by hand:')
        print([1./40, 2./39, 2./37, 2./35])
        print('2nd absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_1[1, :])
        
        # Let x be a continuous random variable following the
        # Laplace distribution of mean 0.0 and scale 2.5. The
        # probability the 1st binary decision is 0 is written
        # p(|x| <= 0.25) = 0.0952. The probability the 2nd binary
        # decision is 0 is written
        # p(0.25 <= |x| <= 0.75)/p(|x| >= 0.25) = 0.1640/0.9048 = 0.1813.
        # The probability the 3rd binary decision is 0 is written
        # p(0.75 <= |x| <= 1.25)/p(|x| >= 0.75) = 0.1343/0.7408 = 0.1813.
        print('Binary probabilities computed by hand:')
        print([0.0952, 0.1813, 0.1813, 0.1813])
        print('3rd absolute centered-quantized latent variable feature map.')
        print('Binary probabilities computed by the function:')
        print(binary_probabilities_1[2, :])
        
        # Let x be a continuous random variable following the
        # standard Cauchy distribution. The probability the 1st
        # binary decision is 0 is written p(|x| <= 0.25) = 0.1560.
        # The probability the 2nd binary decision is 0 is written
        # p(0.25 <= |x| <= 0.75)/p(|x| >= 0.25) = 0.2537/0.8440 = 0.3006.
        # The probability the 3rd binary decision is 0 is written
        # p(0.75 <= |x| <= 1.25)/p(|x| >= 0.75) = 0.1608/0.5903 = 0.2724.
        print('Binary probabilities computed by hand:')
        print([0.1560, 0.3006, 0.2724, 0.2306])
    
    def test_count_binary_decisions(self):
        """Tests the function `count_binary_decisions` in the file "lossless/stats.py".
        
        The test is successful if, for each experiment,
        the number of occurrences of 0 for each binary
        decision computed by the function is equal to the
        number of occurrences of 0 for each binary decision
        computed by hand.
        
        """
        abs_centered_quantized_data_0 = numpy.array([0.75, 0.05, 0.1, 0.2, 0.2, 0.15], dtype=numpy.float32)
        bin_width_test_0 = 0.05
        abs_centered_quantized_data_1 = numpy.array([210., 6., 9., 6.], dtype=numpy.float32)
        bin_width_test_1 = 3.
        truncated_unary_prefix = 7
        
        (cumulated_zeros_0, cumulated_ones_0) = \
            lossless.stats.count_binary_decisions(abs_centered_quantized_data_0,
                                                  bin_width_test_0,
                                                  truncated_unary_prefix)
        (cumulated_zeros_1, cumulated_ones_1) = \
            lossless.stats.count_binary_decisions(abs_centered_quantized_data_1,
                                                  bin_width_test_1,
                                                  truncated_unary_prefix)
        print('1st experiment:')
        print('Number of occurrences of 0 for each binary decision computed by the function:')
        print(cumulated_zeros_0)
        print('Number of occurrences of 0 for each binary decision computed by hand:')
        print(numpy.array([0, 1, 1, 1, 2, 0, 0]))
        print('Number of occurrences of 1 for each binary decision computed by the function:')
        print(cumulated_ones_0)
        print('Number of occurrences of 1 for each binary decision computed by hand:')
        print(numpy.array([6, 5, 4, 3, 1, 1, 1]))
        print('\n2nd experiment:')
        print('Number of occurrences of 0 for each binary decision computed by the function:')
        print(cumulated_zeros_1)
        print('Number of occurrences of 0 for each binary decision computed by hand:')
        print(numpy.array([0, 0, 2, 1, 0, 0, 0]))
        print('Number of occurrences of 1 for each binary decision computed by the function:')
        print(cumulated_ones_1)
        print('Number of occurrences of 1 for each binary decision computed by hand:')
        print(numpy.array([4, 4, 2, 1, 1, 1, 1]))
    
    def test_create_extra(self):
        """Tests the function `create_extra` in the file "lossless/stats.py".
        
        A 1st image is saved at
        "lossless/pseudo_visualization/create_extra/crop_0.png".
        A 2nd image is saved at
        "lossless/pseudo_visualization/create_extra/crop_1.png".
        A 3rd image is saved at
        "lossless/pseudo_visualization/create_extra/crop_2.png".
        The test is successful if each saved
        image is the luminance central crop of
        a different RGB image in the folder
        "lossless/pseudo_data/".
        
        """
        path_to_root = 'lossless/pseudo_data/'
        width_crop = 256
        nb_extra = 3
        path_to_extra = 'lossless/pseudo_data/pseudo_extra.npy'
        
        # The images in the folder "lossless/pseudo_data/"
        # are large. Therefore, none of them is dumped during
        # the preprocessing.
        lossless.stats.create_extra(path_to_root,
                                    width_crop,
                                    nb_extra,
                                    path_to_extra)
        pseudo_extra = numpy.load(path_to_extra)
        for i in range(nb_extra):
            scipy.misc.imsave('lossless/pseudo_visualization/create_extra/crop_{}.png'.format(i),
                              pseudo_extra[i, :, :, 0])
    
    def test_find_index_map_exception(self):
        """Tests the function `find_index_map_exception` in the file "lossless/stats.py".
        
        The test is successful if the index of
        the exception is 3.
        
        """
        nb_images = 100
        height_map = 32
        width_map = 24
        
        y_float32_0 = numpy.random.normal(loc=1.2,
                                          scale=0.5,
                                          size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_1 = numpy.random.laplace(loc=0.4,
                                           scale=1.,
                                           size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_2 = numpy.random.standard_cauchy(size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_3 = numpy.random.uniform(low=-10.,
                                           high=10.,
                                           size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32_4 = numpy.random.normal(loc=-1.,
                                          scale=4.,
                                          size=(nb_images, height_map, width_map, 1)).astype(numpy.float32)
        y_float32 = numpy.concatenate((y_float32_0, y_float32_1, y_float32_2, y_float32_3, y_float32_4),
                                      axis=3)
        idx_map_exception = lossless.stats.find_index_map_exception(y_float32)
        print('Index of the exception: {}'.format(idx_map_exception))
    
    def test_rescale_compress_lossless_maps(self):
        """Tests the function `rescale_compress_lossless_maps` in the file "lossless/compression.py".
        
        The test is successful if, when using the binary probabilities
        saved at "lossless/pseudo_data/binary_probabilities_scale_compress_invalid_0.npy",
        the program raises `RuntimeError` of type 4. This must also
        occur when using the binary probabilities saved at
        "lossless/pseudo_data/binary_probabilities_scale_compress_invalid_1.npy".
        However, when using the binary probabilities saved at
        "lossless/pseudo_data/binary_probabilities_scale_compress_valid.npy",
        the program should not raise any exception.
        
        """
        height_map = 96
        width_map = 48
        bin_widths_test = numpy.array([1.5, 1.5, 1.5], dtype=numpy.float32)
        
        # In "lossless/pseudo_data/binary_probabilities_scale_compress_invalid_0.npy",
        # several binary probabilities are equal to `nan`
        # but the associated binary decisions may occur.
        # In "lossless/pseudo_data/binary_probabilities_scale_compress_invalid_1.npy",
        # several binary probabilities are either negative
        # or larger than 1.
        path_to_binary_probabilities = 'lossless/pseudo_data/binary_probabilities_scale_compress_valid.npy'
        print('The binary probabilities at "{}" are used.'.format(path_to_binary_probabilities))
        
        # The optional argument `loc` of the function
        # `numpy.random.normal` is set to 0.0 as the
        # data must be centered.
        centered_data_0 = numpy.random.normal(loc=0.,
                                              scale=5.,
                                              size=(1, height_map, width_map, 1)).astype(numpy.float32)
        centered_data_1 = numpy.random.normal(loc=0.,
                                              scale=0.2,
                                              size=(1, height_map, width_map, 1)).astype(numpy.float32)
        centered_data_2 = numpy.random.normal(loc=0.,
                                              scale=0.5,
                                              size=(1, height_map, width_map, 1)).astype(numpy.float32)
        centered_data = numpy.concatenate((centered_data_0, centered_data_1, centered_data_2),
                                          axis=3)
        expanded_centered_quantized_data = tls.quantize_per_map(centered_data, bin_widths_test)
        centered_quantized_data = numpy.squeeze(expanded_centered_quantized_data,
                                                axis=0)
        nb_bits = lossless.compression.rescale_compress_lossless_maps(centered_quantized_data,
                                                                      bin_widths_test,
                                                                      path_to_binary_probabilities)
        print('Number of bits in the bitstream: {}'.format(nb_bits))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the libraries in the folder "lossless".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterLossless()
    getattr(tester, 'test_' + args.name)()


