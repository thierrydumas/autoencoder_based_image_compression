"""A script to test the libraries in the folder "eae"."""

import argparse
import numpy
import os
import tensorflow as tf

import eae.analysis
import eae.batching
import eae.graph.components
import eae.graph.constants as csts
import tfutils.tfutils as tfuls
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder
from eae.graph.IsolatedDecoder import IsolatedDecoder


class TesterEntropyAutoencoder(object):
    """Class for testing the libraries in the folder "eae"."""
    
    def test_activate_latent_variables(self):
        """Tests the function `activate_latent_variables` in the file "eae.analysis.py".
        
        The test is successful if the images saved in the directory
        at "eae/pseudo_visualization/activate_latent_variables/" do
        not exhibit any structure.
        
        """
        h_in = 256
        w_in = 384
        idx_map_activation = 4
        tuple_activation_values = (-200., -20., 20., 200.)
        
        isolated_decoder = IsolatedDecoder(1,
                                           h_in,
                                           w_in,
                                           False)
        
        # As the bin widths are not learned, they are all set to 1.
        bin_widths = numpy.ones(csts.NB_MAPS_3, dtype=numpy.float32)
        
        # At initialization, the feature map mean over many
        # luminance images must be close to 0.
        map_mean = numpy.zeros(csts.NB_MAPS_3, dtype=numpy.float32)
        tuple_pairs_row_col = (
            (1, 1),
            (6, 6)
        )
        with tf.Session() as sess:
            isolated_decoder.initialization(sess, '')
            for activation_value in tuple_activation_values:
                path_to_directory_crop = os.path.join('eae/pseudo_visualization/activate_latent_variables/',
                                                      '{0}_{1}'.format(idx_map_activation + 1, tls.float_to_str(activation_value)))
                if not os.path.isdir(path_to_directory_crop):
                    os.makedirs(path_to_directory_crop)
                for (row_activation, col_activation) in tuple_pairs_row_col:
                    eae.analysis.activate_latent_variable(sess,
                                                          isolated_decoder,
                                                          h_in,
                                                          w_in,
                                                          bin_widths,
                                                          row_activation,
                                                          col_activation,
                                                          idx_map_activation,
                                                          activation_value,
                                                          map_mean,
                                                          64,
                                                          64,
                                                          os.path.join(path_to_directory_crop, '{0}_{1}.png'.format(row_activation, col_activation)))
    
    def test_decoder(self):
        """Tests the function `decoder` in the file "eae/graph/components.py".
        
        The test is successful if the 2nd dimension
        of the latent variables perturbed by uniform
        noise is 16 times smaller than the 2nd dimension
        of the reconstruction of the visible units.
        Besides, the 3rd dimension of the latent
        variables perturbed by uniform noise must be
        16 times smaller than the 3rd dimension of the
        reconstruction of the visible units.
        
        """
        are_bin_widths_learned = False
        nb_maps_3 = 6
        nb_maps_2 = 8
        nb_maps_1 = 4
        
        with tf.variable_scope('decoder'):
            if not are_bin_widths_learned:
                gamma_4 = tf.get_variable('gamma_4',
                                          dtype=tf.float32,
                                          initializer=tfuls.initialize_weights_gdn(nb_maps_3, 2.e-5))
                beta_4 = tf.get_variable('beta_4',
                                         dtype=tf.float32,
                                         initializer=tf.ones([nb_maps_3], dtype=tf.float32))
            weights_4 = tf.get_variable('weights_4',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([5, 5, nb_maps_2, nb_maps_3],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_4 = tf.get_variable('biases_4',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([nb_maps_2], dtype=tf.float32))
            gamma_5 = tf.get_variable('gamma_5',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(nb_maps_2, 2.e-5))
            beta_5 = tf.get_variable('beta_5',
                                     dtype=tf.float32,
                                     initializer=tf.ones([nb_maps_2], dtype=tf.float32))
            weights_5 = tf.get_variable('weights_5',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([5, 5, nb_maps_1, nb_maps_2],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_5 = tf.get_variable('biases_5',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([nb_maps_1], dtype=tf.float32))
            gamma_6 = tf.get_variable('gamma_6',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(nb_maps_1, 2.e-5))
            beta_6 = tf.get_variable('beta_6',
                                     dtype=tf.float32,
                                     initializer=tf.ones([nb_maps_1], dtype=tf.float32))
            weights_6 = tf.get_variable('weights_6',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([9, 9, 1, nb_maps_1],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
        node_y_tilde = tf.placeholder(tf.float32, shape=(2, 6, 8, nb_maps_3))
        node_reconstruction = eae.graph.components.decoder(node_y_tilde,
                                                           are_bin_widths_learned)
        print('Shape of the latent variables perturbed by uniform noise:')
        print(node_y_tilde.get_shape())
        print('Shape of the reconstruction of the visible units:')
        print(node_reconstruction.get_shape())
    
    def test_decode_mini_batches(self):
        """Tests the function `decode_mini_batches` in the file "eae/batching.py".
        
        For i = 0 ... 3, an histogram is saved at
        "eae/pseudo_visualization/decode_mini_batches/reconstructed_pixels_i.png".
        The test is successful if, in the first histogram,
        all the values are the same.
        
        """
        batch_size = 2
        h_in = 64
        w_in = 48
        path_to_restore = ''
        
        # 2 batches of quantized latent variables will be created
        # by the function `decode_mini_batches`.
        quantized_y_float32 = numpy.random.randint(
            -6,
            high=6,
            size=(2*batch_size, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, csts.NB_MAPS_3)
        ).astype(numpy.float32)
        quantized_y_float32[0, :, :, :] = 0.
        isolated_decoder = IsolatedDecoder(batch_size,
                                           h_in,
                                           w_in,
                                           False)
        with tf.Session() as sess:
            isolated_decoder.initialization(sess, path_to_restore)
            reconstruction_uint8 = eae.batching.decode_mini_batches(quantized_y_float32,
                                                                    sess,
                                                                    isolated_decoder,
                                                                    batch_size)
        for i in range(reconstruction_uint8.shape[0]):
            tls.histogram(reconstruction_uint8[i, :, :, :].flatten(),
                          'Pixel distribution for the reconstructed image of index {}'.format(i),
                          'eae/pseudo_visualization/decode_mini_batches/reconstructed_pixels_{}.png'.format(i))
    
    def test_encoder(self):
        """Tests the function `encoder` in the file "eae/graph/components.py".
        
        The test is successful if the 2nd dimension
        of the visible units is 16 times larger than
        the 2nd dimension of the latent variables.
        Besides, the 3rd dimension of the visible
        units must be 16 times larger than the 3rd
        dimension of the latent variables.
        
        """
        are_bin_widths_learned = False
        nb_maps_1 = 4
        nb_maps_2 = 8
        nb_maps_3 = 6
        
        with tf.variable_scope('encoder'):
            weights_1 = tf.get_variable('weights_1',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([9, 9, 1, nb_maps_1],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_1 = tf.get_variable('biases_1',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([nb_maps_1], dtype=tf.float32))
            gamma_1 = tf.get_variable('gamma_1',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(nb_maps_1, 2.e-5))
            beta_1 = tf.get_variable('beta_1',
                                     dtype=tf.float32,
                                     initializer=tf.ones([nb_maps_1], dtype=tf.float32))
            weights_2 = tf.get_variable('weights_2',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([5, 5, nb_maps_1, nb_maps_2],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_2 = tf.get_variable('biases_2',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([nb_maps_2], dtype=tf.float32))
            gamma_2 = tf.get_variable('gamma_2',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(nb_maps_2, 2.e-5))
            beta_2 = tf.get_variable('beta_2',
                                     dtype=tf.float32,
                                     initializer=tf.ones([nb_maps_2], dtype=tf.float32))
            weights_3 = tf.get_variable('weights_3',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([5, 5, nb_maps_2, nb_maps_3],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_3 = tf.get_variable('biases_3',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([nb_maps_3], dtype=tf.float32))
            if not are_bin_widths_learned:
                gamma_3 = tf.get_variable('gamma_3',
                                          dtype=tf.float32,
                                          initializer=tfuls.initialize_weights_gdn(nb_maps_3, 2.e-5))
                beta_3 = tf.get_variable('beta_3',
                                         dtype=tf.float32,
                                         initializer=tf.ones([nb_maps_3], dtype=tf.float32))
        node_visible_units = tf.placeholder(tf.float32, shape=(2, 96, 128, 1))
        node_y = eae.graph.components.encoder(node_visible_units,
                                              are_bin_widths_learned)
        print('Shape of the visible units:')
        print(node_visible_units.get_shape())
        print('Shape of the latent variables:')
        print(node_y.get_shape())
    
    def test_encode_mini_batches(self):
        """Tests the function `encode_mini_batches` in the file "eae/batching.py".
        
        For i = 0 ... 3, an histogram is saved at
        "eae/pseudo_visualization/encode_mini_batches/latent_variables_i.png".
        The test is successful if, in the first histogram,
        all the values are around 0. In the third histogram,
        most of the values must be around 0.
        
        """
        batch_size = 2
        h_in = 64
        w_in = 48
        path_to_nb_itvs_per_side_load = ''
        path_to_restore = ''
        
        # 2 batches of luminance images will be created
        # by the function `encode_mini_batches`.
        luminances_uint8 = numpy.random.randint(0,
                                                high=256,
                                                size=(2*batch_size, h_in, w_in, 1),
                                                dtype=numpy.uint8)
        luminances_uint8[0, :, :, :] = 0
        luminances_uint8[1, :, :, :] = 255
        luminances_uint8[2, :, :, :] = 0
        
        # Only a portion of the luminance image of index
        # 2 is white.
        luminances_uint8[2, 0:2, 0:2, :] = 255
        entropy_ae = EntropyAutoencoder(batch_size,
                                        h_in,
                                        w_in,
                                        1.,
                                        12000.,
                                        path_to_nb_itvs_per_side_load,
                                        False)
        with tf.Session() as sess:
            entropy_ae.initialization(sess, path_to_restore)
            y_float32 = eae.batching.encode_mini_batches(luminances_uint8,
                                                         sess,
                                                         entropy_ae,
                                                         batch_size)
        for i in range(luminances_uint8.shape[0]):
            tls.histogram(y_float32[i, :, :, :].flatten(),
                          'Latent variables distribution for the image of index {}'.format(i),
                          'eae/pseudo_visualization/encode_mini_batches/latent_variables_{}.png'.format(i))
    
    def test_fit_maps(self):
        """Tests the function `fit_maps` in the file "eae/analysis.py".
        
        The test is successful if, in the images saved in the
        directory at "eae/pseudo_visualization/fit_maps/", the
        distributions of the feature maps latent variables are
        all similar.
        
        """
        luminance_0_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode('eae/pseudo_data/peppers.png', 'RGB'))[:, :, 0:1]
        luminance_1_uint8 = tls.rgb_to_ycbcr(tls.read_image_mode('eae/pseudo_data/mandrill.png', 'RGB'))[:, :, 0:1]
        luminances_uint8 = numpy.stack((luminance_0_uint8, luminance_1_uint8),
                                       axis=0)
        
        # When the entropy autoencoder model is not a loaded one,
        # the sixth argument of the constructor below is an empty
        # string.
        entropy_ae = EntropyAutoencoder(2,
                                        luminances_uint8.shape[1],
                                        luminances_uint8.shape[2],
                                        1.,
                                        8000.,
                                        '',
                                        False)
        with tf.Session() as sess:
            entropy_ae.initialization(sess, '')
            y_float32 = sess.run(
                entropy_ae.node_y,
                feed_dict={entropy_ae.node_visible_units:luminances_uint8.astype(numpy.float32)}
            )
        eae.analysis.fit_maps(y_float32,
                              'eae/pseudo_visualization/fit_maps/laplace_locations.png',
                              'eae/pseudo_visualization/fit_maps/laplace_scales.png',
                              ['eae/pseudo_visualization/fit_maps/fitting_map_{}.png'.format(i + 1) for i in range(y_float32.shape[3])])
    
    def test_isolated_decoder(self):
        """Tests the class `IsolatedDecoder` in the file "eae/graph/IsolatedDecoder.py".
        
        The test is successful if the 2nd dimension
        of the quantized latent variables is 16 times
        smaller than the 2nd dimension of the reconstruction
        of the visible units. Besides, the 3rd dimension
        of the quantized latent variables must be 16 times
        smaller than the 3rd dimension of the reconstruction
        of the visible units.
        
        """
        batch_size = 40
        h_in = 64
        w_in = 48
        path_to_restore = ''
        
        # The method `__init__` of class `IsolatedDecoder`
        # checks that `h_in` and `w_in` are divisible by
        # `csts.STRIDE_PROD`.
        isolated_decoder = IsolatedDecoder(batch_size,
                                           h_in,
                                           w_in,
                                           False)
        quantized_y_float32 = numpy.random.randint(
            -3,
            high=4,
            size=(batch_size, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, csts.NB_MAPS_3)
        ).astype(numpy.float32)
        with tf.Session() as sess:
            isolated_decoder.initialization(sess, path_to_restore)
            reconstruction_float32 = sess.run(
                isolated_decoder.node_reconstruction,
                feed_dict={isolated_decoder.node_quantized_y:quantized_y_float32}
            )
        print('Shape of the quantized latent variables:')
        print(quantized_y_float32.shape)
        print('Shape of the reconstruction of the visible units:')
        print(reconstruction_float32.shape)    
    
    def test_training_eae_bw(self):
        """Tests the method `training_eae_bw` of class `EntropyAutoencoder` in the file "eae/graph/EntropyAutoencoder.py".
        
        The test is successful if the reconstruction
        error after the training is much smaller than
        the reconstruction error before the training.
        
        """
        batch_size = 40
        h_in = 64
        w_in = 48
        nb_epochs_training = 2000
        path_to_nb_itvs_per_side_load = ''
        path_to_restore = ''
        
        batch_float32 = numpy.random.normal(loc=0.,
                                            scale=20.,
                                            size=(batch_size, h_in, w_in, 1)).astype(numpy.float32)
        entropy_ae = EntropyAutoencoder(batch_size,
                                        h_in,
                                        w_in,
                                        1.,
                                        12000.,
                                        path_to_nb_itvs_per_side_load,
                                        False)
        with tf.Session() as sess:
            entropy_ae.initialization(sess, path_to_restore)
            
            # Here, the number of unit intervals in the
            # right half of the grid before the training
            # can be smaller than the number of unit intervals
            # in the right half of the grid after the training.
            (_, scaled_approx_entropy_0, rec_error_0, _) = entropy_ae.evaluation(sess, batch_float32)
            print('Number of unit intervals in the right half of the grid before the training: {}'.format(entropy_ae.get_nb_intervals_per_side()))
            print('Scaled cumulated approximate entropy before the training: {}'.format(scaled_approx_entropy_0))
            print('Reconstruction error before the training: {}'.format(rec_error_0))
            for _ in range(nb_epochs_training):
                entropy_ae.training_fct(sess, batch_float32)
                entropy_ae.training_eae_bw(sess, batch_float32)
            (_, scaled_approx_entropy_1, rec_error_1, _) = entropy_ae.evaluation(sess, batch_float32)
            print('Number of unit intervals in the right half of the grid after {0} training epochs: {1}'.format(nb_epochs_training, entropy_ae.get_nb_intervals_per_side()))
            print('Scaled cumulated approximate entropy after {0} training epochs: {1}'.format(nb_epochs_training, scaled_approx_entropy_1))
            print('Reconstruction error after {0} training epochs: {1}'.format(nb_epochs_training, rec_error_1))
    
    def test_training_fct(self):
        """Tests the method `training_fct` of class `EntropyAutoencoder` in the file "eae/graph/EntropyAutoencoder.py".
        
        The test is successful if the loss of the
        approximation of unknown probability density
        functions with piecewise linear functions after
        the fitting is smaller than the one before the
        fitting.
        
        """
        batch_size = 40
        h_in = 64
        w_in = 48
        nb_epochs_fitting = 2000
        path_to_nb_itvs_per_side_load = ''
        path_to_restore = ''
        
        batch_float32 = numpy.random.normal(loc=0.,
                                            scale=20.,
                                            size=(batch_size, h_in, w_in, 1)).astype(numpy.float32)
        entropy_ae = EntropyAutoencoder(batch_size,
                                        h_in,
                                        w_in,
                                        1.,
                                        12000.,
                                        path_to_nb_itvs_per_side_load,
                                        False)
        with tf.Session() as sess:
            entropy_ae.initialization(sess, path_to_restore)
            
            # Here, the number of unit intervals in the
            # right half of the grid before the fitting has
            # to be equal to the number of unit intervals
            # in the right half of the grid after the fitting.
            loss_density_approx_0 = entropy_ae.evaluation(sess, batch_float32)[3]
            print('Number of unit intervals in the right half of the grid before the fitting: {}'.format(entropy_ae.get_nb_intervals_per_side()))
            print('Loss of the approximation of unknown probability density functions before the fitting: {}'.format(loss_density_approx_0))
            for _ in range(nb_epochs_fitting):
                entropy_ae.training_fct(sess, batch_float32)
            loss_density_approx_1 = entropy_ae.evaluation(sess, batch_float32)[3]
            print('Number of unit intervals in the right half of the grid after {0} fitting epochs: {1}'.format(nb_epochs_fitting, entropy_ae.get_nb_intervals_per_side()))
            print('Loss of the approximation of unknown probability density functions after {0} fitting epochs: {1}'.format(nb_epochs_fitting, loss_density_approx_1))
    
    def test_weight_l2_norm(self):
        """Tests the function `weight_l2_norm` in the file "eae/graph/components.py".
        
        The test is successful if the cumulated
        weight l2-norm computed by hand is equal
        to the cumulated weight l2-norm computed
        by the function.
        
        """
        with tf.variable_scope('encoder'):
            weights_1 = tf.get_variable('weights_1',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([9, 9, 2, 4], dtype=tf.float32))
            weights_2 = tf.get_variable('weights_2',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([3, 3, 1, 2], dtype=tf.float32))
            weights_3 = tf.get_variable('weights_3',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([3, 3, 1, 2], dtype=tf.float32))
        with tf.variable_scope('decoder'):
            weights_4 = tf.get_variable('weights_4',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([3, 3, 1, 2], dtype=tf.float32))
            weights_5 = tf.get_variable('weights_5',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([3, 3, 1, 2], dtype=tf.float32))
            weights_6 = tf.get_variable('weights_6',
                                        dtype=tf.float32,
                                        initializer=0.5*tf.ones([9, 9, 2, 4], dtype=tf.float32))
        node_weight_decay = eae.graph.components.weight_l2_norm()
        with tf.Session() as sess:
            
            # For details on the condition below, see
            # <https://www.tensorflow.org/api_guides/python/upgrade>.
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
            weight_decay = sess.run(node_weight_decay)
        print('Cumulated weight l2-norm computed by hand: {}'.format(171.))
        print('Cumulated weight l2-norm computed by the function: {}'.format(weight_decay))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the libraries in the folder "eae".')
    parser.add_argument('name', help='name of the function/method to be tested')
    args = parser.parse_args()
    
    tester = TesterEntropyAutoencoder()
    getattr(tester, 'test_' + args.name)()


