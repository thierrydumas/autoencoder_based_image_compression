"""A library that defines the class `EntropyAutoencoder`."""

import numpy
import pickle
import tensorflow as tf

import eae.graph.components
import eae.graph.constants as csts
import tfutils.tfutils as tfuls
import tools.tools as tls


# For big neural networks, a GPU with a large GRAM
# is needed for training. This kind of GPU may be set
# up on a cluster. When using a cluster, breakdowns
# can occur. That is why the training procedure of big
# neural networks has to handle the possible breakdowns.
class EntropyAutoencoder(object):
    """Entropy autoencoder class.
    
    The designation "parameters of the entropy autoencoder"
    refers to all the trainable parameters except the
    parameters of the piecewise linear functions and the
    quantization bin widths.
    
    Attributes
    ----------
    are_bin_widths_learned : bool
        Are the quantization bin widths
        learned?
    All the other attributes are the nodes we
    need to fetch while running the graph.
    
    """
    
    def __init__(self, batch_size, h_in, w_in, bin_width_init, gamma_scaling, path_to_nb_itvs_per_side_load, are_bin_widths_learned):
        """Builds the graph of the entropy autoencoder.
        
        Parameters
        ----------
        batch_size : int
            Size of the mini-batches.
        h_in : int
            Height of the input images.
        w_in : int
            Width of the input images.
        bin_width_init : float
            Value of the quantization bin widths
            at the beginning of the 1st training.
        gamma_scaling : float
            Scaling coefficient. In the objective
            function to be minimized over the
            entropy autoencoder parameters, the
            scaling coefficient weights the cumulated
            approximate entropy of the quantized latent
            variables with respect to the reconstruction
            error and the l2-norm weight decay.
        path_to_nb_itvs_per_side_load : str
            Path to the number of unit intervals
            in the right half of the grid. For the
            1st training, this string is empty.
        are_bin_widths_learned : bool
            Are the quantization bin widths learned?
            If False, the quantization bin widths are
            equal to `bin_width_init` during the training.
        
        Raises
        ------
        ValueError
            If the height of the input images is not divisible
            by the product of the three strides.
        ValueError
            If the width of the input images is not divisible
            by the product of the three strides.
        
        """
        if h_in % csts.STRIDE_PROD != 0:
            raise ValueError('The height of the input images is not divisible by the product of the three strides.')
        if w_in % csts.STRIDE_PROD != 0:
            raise ValueError('The width of the input images is not divisible by the product of the three strides.')
        if path_to_nb_itvs_per_side_load:
            
            # If it is not the 1st training, to build the graph,
            # the number of unit intervals in the right half of
            # the grid is required. This information is
            # stored in a Tensorflow model. But, the model is
            # restored after the creation of the graph. To
            # circumvent this problem, the number of unit
            # intervals in the right half of the grid is stored
            # separately.
            with open(path_to_nb_itvs_per_side_load, 'rb') as file:
                nb_itvs_per_side = pickle.load(file)
        else:
            nb_itvs_per_side = csts.NB_ITVS_PER_SIDE_INIT
        
        # `grid_init` is used as initializer.
        # An initializer must have the same
        # data-type as the tensor it initializes.
        grid_init = numpy.linspace(-nb_itvs_per_side,
                                   nb_itvs_per_side,
                                   num=2*csts.NB_POINTS_PER_INTERVAL*nb_itvs_per_side + 1,
                                   dtype=numpy.float32)
        
        # All variables are defined here as
        # they must be accessible later on
        # to fill in the different lists of
        # variables to be minimized.
        with tf.variable_scope('piecewise_linear_function'):
            bin_widths = tf.get_variable('bin_widths',
                                         dtype=tf.float32,
                                         initializer=bin_width_init*tf.ones([csts.NB_MAPS_3], dtype=tf.float32))
            nb_intervals_per_side = tf.get_variable('nb_intervals_per_side',
                                                    dtype=tf.int64,
                                                    initializer=numpy.array(nb_itvs_per_side, dtype=numpy.int64),
                                                    trainable=False)
            grid = tf.get_variable('grid',
                                   dtype=tf.float32,
                                   initializer=grid_init,
                                   trainable=False)
            
            # If `validate_shape` is set to True,
            # Tensorflow considers that `parameters` has
            # the same shape as its initializer and it
            # builds the gradient computation based
            # on this belief.
            parameters = tf.get_variable('parameters',
                                         dtype=tf.float32,
                                         initializer=tls.tile_cauchy(grid_init, csts.NB_MAPS_3),
                                         validate_shape=False)
        with tf.variable_scope('encoder'):
            weights_1 = tf.get_variable('weights_1',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_1, csts.WIDTH_KERNEL_1, 1, csts.NB_MAPS_1],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
            biases_1 = tf.get_variable('biases_1',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([csts.NB_MAPS_1], dtype=tf.float32))
            gamma_1 = tf.get_variable('gamma_1',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_1, csts.MIN_GAMMA_BETA))
            beta_1 = tf.get_variable('beta_1',
                                     dtype=tf.float32,
                                     initializer=tf.ones([csts.NB_MAPS_1], dtype=tf.float32))
            weights_2 = tf.get_variable('weights_2',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_2, csts.WIDTH_KERNEL_2, csts.NB_MAPS_1, csts.NB_MAPS_2],
                                                                     mean=0.,
                                                                     stddev=0.02,
                                                                     dtype=tf.float32))
            biases_2 = tf.get_variable('biases_2',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([csts.NB_MAPS_2], dtype=tf.float32))
            gamma_2 = tf.get_variable('gamma_2',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_2, csts.MIN_GAMMA_BETA))
            beta_2 = tf.get_variable('beta_2',
                                     dtype=tf.float32,
                                     initializer=tf.ones([csts.NB_MAPS_2], dtype=tf.float32))
            weights_3 = tf.get_variable('weights_3',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_3, csts.WIDTH_KERNEL_3, csts.NB_MAPS_2, csts.NB_MAPS_3],
                                                                     mean=0.,
                                                                     stddev=0.05,
                                                                     dtype=tf.float32))
            biases_3 = tf.get_variable('biases_3',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([csts.NB_MAPS_3], dtype=tf.float32))
            
            # If the quantization bin widths are not learned,
            # an additional normalization is appended to the
            # end of the decoder.
            if not are_bin_widths_learned:
                gamma_3 = tf.get_variable('gamma_3',
                                          dtype=tf.float32,
                                          initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_3, csts.MIN_GAMMA_BETA))
                beta_3 = tf.get_variable('beta_3',
                                         dtype=tf.float32,
                                         initializer=tf.ones([csts.NB_MAPS_3], dtype=tf.float32))
        with tf.variable_scope('decoder'):
            if not are_bin_widths_learned:
                gamma_4 = tf.get_variable('gamma_4',
                                          dtype=tf.float32,
                                          initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_3, csts.MIN_GAMMA_BETA))
                beta_4 = tf.get_variable('beta_4',
                                         dtype=tf.float32,
                                         initializer=tf.ones([csts.NB_MAPS_3], dtype=tf.float32))
            weights_4 = tf.get_variable('weights_4',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_3, csts.WIDTH_KERNEL_3, csts.NB_MAPS_2, csts.NB_MAPS_3],
                                                                     mean=0.,
                                                                     stddev=0.05,
                                                                     dtype=tf.float32))
            biases_4 = tf.get_variable('biases_4',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([csts.NB_MAPS_2], dtype=tf.float32))
            gamma_5 = tf.get_variable('gamma_5',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_2, csts.MIN_GAMMA_BETA))
            beta_5 = tf.get_variable('beta_5',
                                     dtype=tf.float32,
                                     initializer=tf.ones([csts.NB_MAPS_2], dtype=tf.float32))
            weights_5 = tf.get_variable('weights_5',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_2, csts.WIDTH_KERNEL_2, csts.NB_MAPS_1, csts.NB_MAPS_2],
                                                                     mean=0.,
                                                                     stddev=0.02,
                                                                     dtype=tf.float32))
            biases_5 = tf.get_variable('biases_5',
                                       dtype=tf.float32,
                                       initializer=tf.zeros([csts.NB_MAPS_1], dtype=tf.float32))
            gamma_6 = tf.get_variable('gamma_6',
                                      dtype=tf.float32,
                                      initializer=tfuls.initialize_weights_gdn(csts.NB_MAPS_1, csts.MIN_GAMMA_BETA))
            beta_6 = tf.get_variable('beta_6',
                                     dtype=tf.float32,
                                     initializer=tf.ones([csts.NB_MAPS_1], dtype=tf.float32))
            weights_6 = tf.get_variable('weights_6',
                                        dtype=tf.float32,
                                        initializer=tf.random_normal([csts.WIDTH_KERNEL_1, csts.WIDTH_KERNEL_1, 1, csts.NB_MAPS_1],
                                                                     mean=0.,
                                                                     stddev=0.01,
                                                                     dtype=tf.float32))
        
        with tf.variable_scope('decaying_lr'):
            global_step = tf.get_variable('global_step',
                                          dtype=tf.int32,
                                          initializer=0,
                                          trainable=False)
        
        # The duration of the training must be
        # extended when `gamma_scaling` is relatively
        # small.
        if gamma_scaling < 60000.:
            boundaries = [1500000, 2000000]
        elif gamma_scaling >= 60000. and gamma_scaling < 80000.:
            boundaries = [900000, 950000]
        else:
            boundaries = [750000, 800000]
        self.node_lr_eae = tf.train.piecewise_constant(global_step,
                                                       boundaries,
                                                       [csts.LR_EAE, 0.1*csts.LR_EAE, 0.01*csts.LR_EAE])
        
        # Below is the generation of the latent
        # variables perturbed by uniform noise
        # from the visible units.
        self.node_visible_units = tf.placeholder(tf.float32,
                                                 shape=(batch_size, h_in, w_in, 1))
        self.node_y = eae.graph.components.encoder(self.node_visible_units,
                                                   are_bin_widths_learned)
        self.node_y_tilde = tfuls.add_noise(self.node_y, bin_widths)
        node_y_tilde_2d = tfuls.reshape_4d_to_2d(self.node_y_tilde)
        node_approximate_prob = tfuls.approximate_probability(node_y_tilde_2d,
                                                              parameters,
                                                              csts.NB_POINTS_PER_INTERVAL,
                                                              nb_intervals_per_side)
        
        # The 6th argument of the function `tfuls.expand_all`
        # is the largest absolute latent variable plus half the
        # largest quantization bin width.
        (grid_exp, parameters_exp, nb_intervals_per_side_exp) = \
            tfuls.expand_all(grid,
                             parameters,
                             csts.LOW_PROJECTION,
                             csts.NB_POINTS_PER_INTERVAL,
                             nb_intervals_per_side,
                             tf.reduce_max(tf.abs(self.node_y)) + 0.5*tf.reduce_max(bin_widths))
        
        # The operation `tf.assign` outputs a tensor holding
        # the new value of its first argument after the assignment.
        self.node_expansion = [
            tf.assign(grid, grid_exp, validate_shape=False),
            tf.assign(parameters, parameters_exp, validate_shape=False),
            tf.assign(nb_intervals_per_side, nb_intervals_per_side_exp)
        ]
        
        # Below is the training of the parameters of
        # the piecewise linear functions.
        self.node_loss_density_approx = tfuls.loss_density_approximation(node_approximate_prob,
                                                                         parameters,
                                                                         csts.NB_POINTS_PER_INTERVAL)
        self.node_opt_fct = tf.train.GradientDescentOptimizer(learning_rate=csts.LR_FCT).minimize(
            self.node_loss_density_approx,
            var_list=[parameters]
        )
        
        # Below is the projection of the parameters
        # of the piecewise linear functions.
        self.node_projection_parameters_fct = tf.assign(
            parameters,
            tf.maximum(parameters, csts.LOW_PROJECTION)
        )
        
        # Below is the computation of the area
        # under each piecewise linear function.
        self.node_area = tfuls.area_under_piecewise_linear_functions(parameters,
                                                                     csts.NB_POINTS_PER_INTERVAL,
                                                                     nb_intervals_per_side)
        
        # Below is the reconstruction of the visible units from
        # the latent variables perturbed by uniform noise.
        self.node_reconstruction = eae.graph.components.decoder(self.node_y_tilde,
                                                                are_bin_widths_learned)
        
        # Below is the training of the parameters of the
        # entropy autoencoder and the quantization bin widths.
        self.node_rec_error = tfuls.reconstruction_error(self.node_visible_units,
                                                         self.node_reconstruction)
        self.node_scaled_approx_entropy = gamma_scaling*tfuls.approximate_entropy(node_approximate_prob,
                                                                                  bin_widths)
        self.node_weight_decay = csts.WEIGHT_DECAY_P*eae.graph.components.weight_l2_norm()
        loss_eae_bw = self.node_rec_error + self.node_scaled_approx_entropy + self.node_weight_decay
        var_list_eae = [
            weights_1,
            biases_1,
            gamma_1,
            beta_1,
            weights_2,
            biases_2,
            gamma_2,
            beta_2,
            weights_3,
            biases_3,
            weights_4,
            biases_4,
            gamma_5,
            beta_5,
            weights_5,
            biases_5,
            gamma_6,
            beta_6,
            weights_6
        ]
        if not are_bin_widths_learned:
            var_list_eae.append(gamma_3)
            var_list_eae.append(beta_3)
            var_list_eae.append(gamma_4)
            var_list_eae.append(beta_4)
        self.node_opt_eae = tf.train.AdamOptimizer(learning_rate=self.node_lr_eae).minimize(
            loss_eae_bw,
            var_list=var_list_eae,
            global_step=global_step
        )
        self.node_opt_bw = tf.train.GradientDescentOptimizer(learning_rate=csts.LR_BW).minimize(
            loss_eae_bw,
            var_list=[bin_widths]
        )
        
        # Below is the projection of the weights
        # of all GDNs and IGDNs.
        self.node_projection_beta = [
            tf.assign(beta_1, tf.maximum(beta_1, csts.MIN_GAMMA_BETA)),
            tf.assign(beta_2, tf.maximum(beta_2, csts.MIN_GAMMA_BETA)),
            tf.assign(beta_5, tf.maximum(beta_5, csts.MIN_GAMMA_BETA)),
            tf.assign(beta_6, tf.maximum(beta_6, csts.MIN_GAMMA_BETA))
        ]
        
        # Below is the projection of the additive coefficients
        # of all GDNs and IGDNs.
        self.node_projection_gamma = [
            tf.assign(gamma_1, tf.maximum(gamma_1, csts.MIN_GAMMA_BETA)),
            tf.assign(gamma_2, tf.maximum(gamma_2, csts.MIN_GAMMA_BETA)),
            tf.assign(gamma_5, tf.maximum(gamma_5, csts.MIN_GAMMA_BETA)),
            tf.assign(gamma_6, tf.maximum(gamma_6, csts.MIN_GAMMA_BETA))
        ]
    
        # Below is the symmetrization of the
        # weights of all GDNs and IGDNs.
        self.node_symmetric_gamma = [
            tf.assign(gamma_1, 0.5*(gamma_1 + tf.transpose(gamma_1))),
            tf.assign(gamma_2, 0.5*(gamma_2 + tf.transpose(gamma_2))),
            tf.assign(gamma_5, 0.5*(gamma_5 + tf.transpose(gamma_5))),
            tf.assign(gamma_6, 0.5*(gamma_6 + tf.transpose(gamma_6)))
        ]
        if not are_bin_widths_learned:
            self.node_projection_beta.append(tf.assign(beta_3, tf.maximum(beta_3, csts.MIN_GAMMA_BETA)))
            self.node_projection_beta.append(tf.assign(beta_4, tf.maximum(beta_4, csts.MIN_GAMMA_BETA)))
            self.node_projection_gamma.append(tf.assign(gamma_3, tf.maximum(gamma_3, csts.MIN_GAMMA_BETA)))
            self.node_projection_gamma.append(tf.assign(gamma_4, tf.maximum(gamma_4, csts.MIN_GAMMA_BETA)))
            self.node_symmetric_gamma.append(tf.assign(gamma_3, 0.5*(gamma_3 + tf.transpose(gamma_3))))
            self.node_symmetric_gamma.append(tf.assign(gamma_4, 0.5*(gamma_4 + tf.transpose(gamma_4))))
        
        # Below is the projection of the quantization
        # bin widths.
        self.node_projection_bw = tf.assign(bin_widths,
                                            tf.clip_by_value(bin_widths, csts.MIN_BW, csts.MAX_BW))
        
        # Below is the backup of all variables.
        self.node_saver = tf.train.Saver()
        
        # The method `training_eae_bw` needs to know
        # whether the quantization bin widths are learned.
        # This information is conveyed via the attribute
        # `self.are_bin_widths_learned`.
        self.are_bin_widths_learned = are_bin_widths_learned
    
    def get_bin_widths(self):
        """Returns the quantization bin widths.
        
        Returns
        -------
        numpy.ndarray
            1D array with data-type `numpy.float32`.
            Quantization bin widths.
        
        """
        with tf.variable_scope('piecewise_linear_function', reuse=True):
            return tf.get_variable('bin_widths', dtype=tf.float32).eval()
    
    def get_global_step(self):
        """Returns the number of updates of the parameters of the entropy autoencoder.
        
        Returns
        -------
        numpy.ndarray
            0D array with data-type `numpy.int32`.
            Number of updates of the parameters of
            the entropy autoencoder since the
            beginning of the 1st training.
        
        """
        with tf.variable_scope('decaying_lr', reuse=True):
            return tf.get_variable('global_step', dtype=tf.int32).eval()
    
    def get_nb_intervals_per_side(self):
        """Returns the number of unit intervals in the right half of the grid.
        
        Returns
        -------
        numpy.ndarray
            0D array with data-type `numpy.int64`.
            Number of unit intervals in the right half
            of the grid. The grid is symmetrical about 0.
        
        """
        with tf.variable_scope('piecewise_linear_function', reuse=True):
            return tf.get_variable('nb_intervals_per_side', dtype=tf.int64).eval()
    
    def initialization(self, sess, path_to_restore):
        """Either initializes all variables or restores a previous model.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        path_to_restore : str
            Path to a previous model. If
            it is an empty string, all
            variables are initialized. The
            path ends with ".ckpt".
        
        """
        if path_to_restore:
            
            # When the variables are restored from a file,
            # we do not have to initialize them beforehand.
            self.node_saver.restore(sess, path_to_restore)
        else:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()
    
    def save(self, sess, path_to_model, path_to_nb_itvs_per_side_save):
        """Saves the model using the Tensorflow protocol.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        path_to_model : str
            Path to the saved Tensorflow model.
            The path ends with ".ckpt".
        path_to_nb_itvs_per_side_save : str
            Path to the number of unit intervals
            in the right half of the grid.
        
        """
        self.node_saver.save(sess, path_to_model)
        with open(path_to_nb_itvs_per_side_save, 'wb') as file:
            pickle.dump(self.get_nb_intervals_per_side().item(), file, protocol=2)
    
    def training_fct(self, sess, visible_units):
        """"Trains the parameters of the piecewise linear functions.
        
        The grid and the parameters of the piecewise
        linear functions are expanded if the condition
        of expansion is met. Then, the parameters of the
        piecewise linear functions are updated one time.
        Finally, the parameters of the piecewise linear
        functions are projected.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        visible_units : numpy.ndarray
            4D array with data-type `numpy.float32`.
            Visible units. The 4th dimension of
            `visible_units` is equal to 1.
        
        """
        sess.run(self.node_expansion, feed_dict={self.node_visible_units:visible_units})
        sess.run(self.node_opt_fct, feed_dict={self.node_visible_units:visible_units})
        sess.run(self.node_projection_parameters_fct)
    
    def training_eae_bw(self, sess, visible_units):
        """"Trains the parameters of the entropy autoencoder and the quantization bin widths if required.
        
        The parameters of the entropy autoencoder are updated
        one time. Then, the weights of all GDNs/IGDNs and the
        additive coefficients of all GDNs/IGDNs are projected.
        The weights of all GDNs/IGDNs are also symmetrized. If
        required, the quantization bin widths are updated one
        time and they are projected.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        visible_units : numpy.ndarray
            4D array with data-type `numpy.float32`.
            Visible units. The 4th dimension of
            `visible_units` is equal to 1.
        
        """
        if self.are_bin_widths_learned:
            
            # If `self.node_opt_eae` and `self.node_opt_bw` were
            # fetched via two separate graph runs, we would have
            # to check whether the condition of expansion is met
            # between the two separate runs.
            sess.run([self.node_opt_eae, self.node_opt_bw], feed_dict={self.node_visible_units:visible_units})
            sess.run(self.node_projection_bw)
        else:
            sess.run(self.node_opt_eae, feed_dict={self.node_visible_units:visible_units})
        sess.run(self.node_projection_beta)
        sess.run(self.node_projection_gamma)
        sess.run(self.node_symmetric_gamma)
    
    def evaluation(self, sess, visible_units):
        """Computes 4 indicators to assess how the training advances.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        visible_units : numpy.ndarray
            4D array with data-type `tf.float32`.
            Visible units. The 4th dimension of
            `visible_units` is equal to 1.
        
        Returns
        -------
        tuple
            numpy.float64
                Mean entropy of the quantized
                latent variables.
            numpy.float32
                Cumulated approximate entropy
                of the quantized latent variables
                multiplied by the scaling
                coefficient.
            numpy.float32
                Error between the visible units
                and their reconstruction.
            numpy.float32
                Loss of the approximation of the
                probability density functions of
                the latent variables perturbed by
                uniform noise with piecewise linear
                functions.
        
        """
        # When feeding `visible_units` into the
        # graph of the entropy autoencoder, is
        # the condition of expansion met?
        sess.run(self.node_expansion, feed_dict={self.node_visible_units:visible_units})
        scaled_approx_entropy, rec_error, loss_density_approx, y = sess.run(
            [self.node_scaled_approx_entropy, self.node_rec_error, self.node_loss_density_approx, self.node_y],
            feed_dict={self.node_visible_units:visible_units}
        )
        
        # The quantization bin widths may
        # evolve during the training. That is why
        # their current state is retrieved.
        mean_disc_entropy = tls.average_entropies(y, self.get_bin_widths())
        return (mean_disc_entropy, scaled_approx_entropy, rec_error, loss_density_approx)
    
    def checking_activations_1(self, sess, visible_units, titles, paths):
        """Creates the normed histogram of each feature map in the latent variables perturbed by uniform noise.
        
        After creating the normed histogram of each
        feature map in the latent variables perturbed
        by uniform noise, the normed histograms are saved.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        visible_units : numpy.ndarray
            4D array with data-type `tf.float32`.
            Visible units. The 4th dimension of
            `visible_units` is equal to 1.
        titles : list
            The ith string in this list is the title
            of the ith normed histogram. In order to
            consider the first `n` feature maps instead
            of all of them, fill `titles` so that
            `len(titles)` is equal to `n`.
        paths : list
            The ith string in this list is the
            path to the ith saved normed histogram.
            Each path ends with ".png". `len(paths)`
            is equal to `len(titles)`.
        
        """
        with tf.variable_scope('piecewise_linear_function', reuse=True):
            grid = tf.get_variable('grid', dtype=tf.float32).eval()
            parameters = tf.get_variable('parameters', dtype=tf.float32).eval()
        y_tilde = sess.run(self.node_y_tilde, feed_dict={self.node_visible_units:visible_units})
        tls.normed_histogram(y_tilde[:, :, :, 0:len(titles)],
                             grid,
                             parameters[0:len(titles), :],
                             titles,
                             paths)
    
    def checking_activations_2(self, sess, visible_units, paths):
        """Arranges the latent variable feature maps in images and saves the images.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        visible_units : numpy.ndarray
            4D array with data-type `tf.float32`.
            Visible units.  The 4th dimension of
            `visible_units` is equal to 1. The
            latent variable feature maps computed from
            `visible_units[i, :, :, :]` are arranged
            in the ith image.
        paths : list
            The ith string in this list is the path
            to the ith saved image. Each path ends
            with ".png".
        
        """
        y = sess.run(self.node_y, feed_dict={self.node_visible_units:visible_units})
        for i in range(len(paths)):
            tls.visualize_representation(y[i, :, :, :],
                                         8,
                                         paths[i])
    
    def checking_area_under_piecewise_linear_functions(self, sess, title, path):
        """Creates the histogram of the areas under the piecewise linear functions and saves the histogram.
        
        Parameters
        ----------
        sess : Session
            Session that runs the graph.
        title : str
            Title of the histogram.
        path : str
            Path to the saved histogram. The
            path ends with ".png".
        
        """
        area = sess.run(self.node_area)
        tls.histogram(area,
                      title,
                      path)
    
    def checking_p_1(self, str_scope, str_variable, title, path):
        """Creates the histogram of a variable and saves the histogram.
        
        Parameters
        ----------
        str_scope : str
            Scope of the variable.
        str_variable : str
            Name of the variable.
        title : str
            Title of the histogram.
        path : str
            Path to the saved histogram. The
            path ends with ".png".
        
        """
        with tf.variable_scope(str_scope, reuse=True):
            variable = tf.get_variable(str_variable, dtype=tf.float32).eval()
        tls.histogram(variable.flatten(),
                      title,
                      path)
    
    def checking_p_2(self, is_encoder, nb_vertically, path):
        """Arranges the weight filters in a single image and saves the single image.
        
        Parameters
        ----------
        is_encoder : bool
            If True, the weight filters in the 1st
            encoder layer are considered. Otherwise,
            the weight filters in the 3rd decoder
            layer are considered.
        nb_vertically : int
            Number of weight filters per column
            in the single image.
        path : str
            Path to the saved single image. The
            path ends with ".png".
        
        """
        if is_encoder:
            with tf.variable_scope('encoder', reuse=True):
                weights = tf.get_variable('weights_1', dtype=tf.float32).eval()
        else:
            with tf.variable_scope('decoder', reuse=True):
                weights = tf.get_variable('weights_6', dtype=tf.float32).eval()
        tls.visualize_weights(weights,
                              nb_vertically,
                              path)
    
    def checking_p_3(self, str_scope, index_gdn, path):
        """Saves the image of the GDN/IGDN weights.
        
        Parameters
        ----------
        str_scope : str
            Scope of the GDN/IGDN weights.
        index_gdn : int
            Index of the GDN/IGDN weights. The
            index belongs to [|1, 6|].
        path : str
            Path to the saved image. The path
            ends with ".png".
        
        """
        with tf.variable_scope(str_scope, reuse=True):
            gamma_float32 = tf.get_variable('gamma_{}'.format(index_gdn), dtype=tf.float32).eval()
        minimum = numpy.amin(gamma_float32)
        maximum = numpy.amax(gamma_float32)
        gamma_uint8 = numpy.round(255.*(gamma_float32 - minimum)/(maximum - minimum)).astype(numpy.uint8)
        tls.save_image(path,
                       gamma_uint8)


