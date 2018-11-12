"""A library that defines the class `EntropyAutoencoder`."""

import numpy
import os
import scipy.stats.distributions
import warnings

import tools.tools as tls


class EntropyAutoencoder(object):
    """Entropy autoencoder class.
    
    The term "parameters of the entropy autoencoder"
    refers to all the trainable parameters except the
    parameters of the piecewise linear function and the
    quantization bin width.
    
    The eight attributes below are not written as
    conventionally private attributes for readability.
    Yet, they must never be set manually.
    
    Attributes
    ----------
    nb_visible : int
        Number of visible units.
    nb_hidden : int
        Number of encoder hidden units.
    nb_y : int
        Number of latent variables.
    bin_width : float
        Quantization bin width.
    gamma : float
        Scaling coefficient. In the objective
        function to be minimized over the entropy
        autoencoder parameters, the scaling
        coefficient weights the approximate
        entropy of the quantized latent variables
        with respect to the reconstruction error
        and the l2-norm weight decay.
    is_bin_width_learned : bool
        Is the quantization bin width learned?
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid.
    
    All the other attributes are conventionally private.
    
    """
    
    def __init__(self, nb_visible, nb_hidden, nb_y, bin_width_init, gamma, is_bin_width_learned,
                 nb_points_per_interval=4, nb_intervals_per_side=10, low_projection=1.e-6,
                 lr_eae=4.e-5, momentum_eae=0.9, lr_fct=0.2, lr_bw=1.e-5, weights_decay_p=5.e-4):
        """Initializes the entropy autoencoder.
        
        This initialization includes the initialization
        of the parameters of the entropy autoencoder,
        i.e weights and biases, the initialization of the
        parameters of the piecewise linear function and the
        initialization of the quantization bin width.
        It also includes the initialization of the training
        hyperparameters.
        
        Parameters
        ----------
        nb_visible : int
            Number of visible units.
        nb_hidden : int
            Number of encoder hidden units.
        nb_y : int
            Number of latent variables.
        bin_width_init : float
            Value of the quantization bin width
            at the beginning of the training.
        gamma : float
            Scaling coefficient.
        is_bin_width_learned : bool
            Is the quantization bin width learned?
            If False, the quantization bin width never
            changes during the training.
        nb_points_per_interval : int, optional
            Number of sampling points per unit interval in
            the grid. The default value is 4.
        nb_intervals_per_side : int, optional
            Number of unit intervals in the right half of
            the grid. The grid is symmetrical about 0. The
            default value is 10.
        low_projection : float, optional
            Strictly positive minimum for the parameters
            of the piecewise linear function. Thanks to
            `low_projection`, the parameters of the piecewise
            linear function cannot get extremely close to 0.
            Therefore, the limited floating-point precision
            cannot round them to 0. The default value is 1.0e-6.
        lr_eae : float, optional
            Learning rate for the parameters of the entropy
            autoencoder. The default value is 4.0e-5.
        momentum_eae : float, optional
            Momentum for the parameters of the entropy
            autoencoder. The default value is 0.9.
        lr_fct : float, optional
            Learning rate for the parameters of the piecewise
            linear function. The default value is 0.2.
        lr_bw : float, optional
            Learning rate for the quantization bin width.
            The default value is 1.0e-5.
        weights_decay_p : float, optional
            Coefficient that weights the l2-norm weight
            decay with respect to the scaled approximate
            entropy of the quantized latent variables and
            the reconstruction error in the objective
            function to be minimized over the entropy
            autoencoder parameters. The default value is
            5.0e-4.
        
        """
        self.nb_visible = nb_visible
        self.nb_hidden = nb_hidden
        self.nb_y = nb_y
        self.bin_width = bin_width_init
        self.gamma = gamma
        self.is_bin_width_learned = is_bin_width_learned
        
        # `self.nb_points_per_interval` is never
        # modified whereas `self.nb_intervals_per_side`
        # can be modified by the method `__checking_grid`.
        self.nb_points_per_interval = nb_points_per_interval
        self.nb_intervals_per_side = nb_intervals_per_side
        self.__grid = numpy.linspace(-nb_intervals_per_side,
                                     nb_intervals_per_side,
                                     num=2*nb_points_per_interval*nb_intervals_per_side + 1)
        self.__low_projection = low_projection
        self.__parameters_fct = numpy.maximum(scipy.stats.distributions.cauchy.pdf(self.__grid),
                                              self.__low_projection)
        self.__parameters_eae = self.__initialize_parameters_eae()
        self.__updates_eae = self.__initialize_updates_eae()
        self.__lr_eae = lr_eae
        self.__momentum_eae = momentum_eae
        self.__lr_fct = lr_fct
        self.__lr_bw = lr_bw
        self.__weights_decay_p = weights_decay_p
    
    def __initialize_parameters_eae(self):
        """Initializes the parameters of the entropy autoencoder.
        
        Returns
        -------
        dict
            Parameters of the entropy autoencoder.
        
        """
        parameters_eae = dict()
        parameters_eae['weights_encoder'] = {
            'l1': numpy.random.normal(loc=0.,
                                      scale=0.01,
                                      size=(self.nb_visible, self.nb_hidden)),
            'latent': numpy.random.normal(loc=0.,
                                          scale=0.05,
                                          size=(self.nb_hidden, self.nb_y))
        }
        parameters_eae['biases_encoder'] = {
            'l1': numpy.zeros((1, self.nb_hidden)),
            'latent': numpy.zeros((1, self.nb_y))
        }
        parameters_eae['weights_decoder'] = {
            'l1': numpy.random.normal(loc=0.,
                                      scale=0.05,
                                      size=(self.nb_y, self.nb_hidden)),
            'mean': numpy.random.normal(loc=0.,
                                        scale=0.01,
                                        size=(self.nb_hidden, self.nb_visible))
        }
        parameters_eae['biases_decoder'] = {
            'l1': numpy.zeros((1, self.nb_hidden)),
            'mean': numpy.zeros((1, self.nb_visible))
        }
        return parameters_eae
    
    def __initialize_updates_eae(self):
        """Initializes the updates of the parameters of the entropy autoencoder.
        
        Returns
        -------
        dict
            Updates of the parameters of the
            entropy autoencoder.
        
        """
        updates_eae = dict()
        updates_eae['weights_encoder'] = {
            'l1': numpy.zeros((self.nb_visible, self.nb_hidden)),
            'latent': numpy.zeros((self.nb_hidden, self.nb_y))
        }
        updates_eae['biases_encoder'] = {
            'l1': numpy.zeros((1, self.nb_hidden)),
            'latent': numpy.zeros((1, self.nb_y))
        }
        updates_eae['weights_decoder'] = {
            'l1': numpy.zeros((self.nb_y, self.nb_hidden)),
            'mean': numpy.zeros((self.nb_hidden, self.nb_visible))
        }
        updates_eae['biases_decoder'] = {
            'l1': numpy.zeros((1, self.nb_hidden)),
            'mean': numpy.zeros((1, self.nb_visible))
        }
        
        # Note that the entropy autoencoder
        # parameters stored in the dictionary
        # named `self.__parameters_eae` and their
        # corresponding updates stored in the
        # dictionary named `updates_eae` are laid
        # out in a similar manner.
        return updates_eae
    
    def encoder(self, visible_units):
        """Computes the neural activations in the encoder.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
            
        Returns
        -------
        tuple
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Encoder hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Latent variables.
        
        """
        nb_examples = visible_units.shape[0]
        hidden_encoder = tls.leaky_relu(numpy.dot(visible_units,
            self.__parameters_eae['weights_encoder']['l1']) +
            numpy.tile(self.__parameters_eae['biases_encoder']['l1'],
            (nb_examples, 1)))
        y = numpy.dot(hidden_encoder,
            self.__parameters_eae['weights_encoder']['latent']) + \
            numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
            (nb_examples, 1))
        return (hidden_encoder, y)
    
    def decoder(self, y_tilde):
        """Computes the neural activations in the decoder.
        
        Parameters
        ----------
        y_tilde : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables perturbed by uniform noise.
            
        Returns
        -------
        tuple
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Decoder hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Reconstruction of the visible units.
        
        """
        nb_examples = y_tilde.shape[0]
        hidden_decoder = tls.leaky_relu(numpy.dot(y_tilde,
            self.__parameters_eae['weights_decoder']['l1']) +
            numpy.tile(self.__parameters_eae['biases_decoder']['l1'],
            (nb_examples, 1)))
        reconstruction = numpy.dot(hidden_decoder,
            self.__parameters_eae['weights_decoder']['mean']) + \
            numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
            (nb_examples, 1))
        return (hidden_decoder, reconstruction)
    
    def forward_pass(self, visible_units, eps):
        """Computes the neural activations in the entropy autoencoder.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Noise from the uniform distribution of support
            [`-0.5*self.bin_width`, `0.5*self.bin_width`].
        
        Returns
        -------
        tuple
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Encoder hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Latent variables.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Latent variables perturbed by uniform noise.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Decoder hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Reconstruction of the visible units.
        
        """
        (hidden_encoder, y) = self.encoder(visible_units)
        y_tilde = y + eps
        (hidden_decoder, reconstruction) = self.decoder(y_tilde)
        return (hidden_encoder, y, y_tilde, hidden_decoder, reconstruction)
    
    def __checking_gfct(self, flattened_y_tilde, gradients_fct, path_to_checking_g):
        """Runs gradient checking for the parameters of the piecewise linear function.
        
        Parameters
        ----------
        flattened_y_tilde : numpy.ndarray
            1D array with data-type `numpy.float64`.
            Latent variables perturbed by uniform noise.
        gradients_fct : numpy.ndarray
            1D array with data-type `numpy.float64`.
            Gradients of the parameters of the piecewise
            linear function.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        nb_points = gradients_fct.size
        approx = numpy.zeros(nb_points)
        offset = 1.e-4
        for i in range(nb_points):
            pos = self.__parameters_fct.copy()
            pos[i] += offset
            loss_pos = tls.loss_density_approximation(flattened_y_tilde,
                                                      pos,
                                                      self.nb_points_per_interval,
                                                      self.nb_intervals_per_side)
            neg = self.__parameters_fct.copy()
            neg[i] -= offset
            loss_neg = tls.loss_density_approximation(flattened_y_tilde,
                                                      neg,
                                                      self.nb_points_per_interval,
                                                      self.nb_intervals_per_side)
            approx[i] = 0.5*(loss_pos - loss_neg)/offset
        
        tls.histogram(gradients_fct - approx,
                      'Gradient checking for the parameters of the piecewise linear function',
                      os.path.join(path_to_checking_g, 'gfct.png'))
    
    def __checking_gw_4(self, visible_units, y_tilde, hidden_decoder, gwd_mean, path_to_checking_g):
        """Runs gradient checking for the 4th set of weights.
        
        The 4th set of weights gathers the weights
        connecting the decoder hidden units to the
        visible units reconstruction.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        y_tilde : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables perturbed by uniform noise.
        hidden decoder : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Decoder hidden units.
        gwd_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            4th set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.nb_hidden, self.nb_visible))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        
        # The dictionary containing all the parameters
        # should not be modified by the methods running
        # gradient checking.
        # The l2-norm weight decay contains some terms
        # that do not cancel out during gradient checking.
        for i in range(self.nb_hidden):
            for j in range(self.nb_visible):
                pos = self.__parameters_eae['weights_decoder']['mean'].copy()
                pos[i, j] += offset
                reconstruction = numpy.dot(hidden_decoder, pos) + \
                    numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                    (nb_examples, 1))
                loss_pos = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_pos = loss_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters_eae['weights_decoder']['mean'].copy()
                neg[i, j] -= offset
                reconstruction = numpy.dot(hidden_decoder, neg) + \
                    numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                    (nb_examples, 1))
                loss_neg = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_neg = loss_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
        
        tls.histogram((gwd_mean - approx).flatten(),
                      'Gradient checking weights (4)',
                      os.path.join(path_to_checking_g, 'gw_4.png'))
    
    def __checking_gw_3(self, visible_units, y_tilde, gwd_l1, path_to_checking_g):
        """Runs gradient checking for the 3rd set of weights.
        
        The 3rd set of weights gathers the weights
        connecting the latent variables perturbed by
        uniform noise to the decoder hidden units.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        y_tilde : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables perturbed by uniform noise.
        gwd_l1 : numpy.ndarray
            2D array with data-type `numpy.float64`.
            3rd set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.nb_y, self.nb_hidden))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_lrelu = False
        for i in range(self.nb_y):
            for j in range(self.nb_hidden):
                pos = self.__parameters_eae['weights_decoder']['l1'].copy()
                pos[i, j] += offset
                hidden_decoder_pos = tls.leaky_relu(numpy.dot(y_tilde, pos) +
                    numpy.tile(self.__parameters_eae['biases_decoder']['l1'],
                    (nb_examples, 1)))
                reconstruction = numpy.dot(hidden_decoder_pos,
                    self.__parameters_eae['weights_decoder']['mean']) + \
                    numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                    (nb_examples, 1))
                loss_pos = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_pos = loss_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters_eae['weights_decoder']['l1'].copy()
                neg[i, j] -= offset
                hidden_decoder_neg = tls.leaky_relu(numpy.dot(y_tilde, neg) +
                    numpy.tile(self.__parameters_eae['biases_decoder']['l1'],
                    (nb_examples, 1)))
                reconstruction = numpy.dot(hidden_decoder_neg,
                    self.__parameters_eae['weights_decoder']['mean']) + \
                    numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                    (nb_examples, 1))
                loss_neg = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_neg = loss_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                if numpy.any(numpy.sign(hidden_decoder_pos) != numpy.sign(hidden_decoder_neg)):
                    is_non_diff_lrelu = True
        
        # If Leaky ReLU wrecks the gradient checking,
        # the difference between the gradient and its
        # finite difference approximation is not saved
        # as it is not interpretable.
        if is_non_diff_lrelu:
            warnings.warn('Leaky ReLU wrecks the gradient checking of the 3rd set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwd_l1 - approx).flatten(),
                          'Gradient checking weights (3)',
                          os.path.join(path_to_checking_g, 'gw_3.png'))
    
    def __checking_gb_3(self, visible_units, y_tilde, gbd_l1, path_to_checking_g):
        """Runs gradient checking for the 3rd set of biases.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        y_tilde : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables perturbed by uniform noise.
        gbd_l1 : numpy.ndarray
            1D array with data-type `numpy.float64`.
            3rd set of bias gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros(self.nb_hidden)
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_lrelu = False
        
        # The biases in the 3rd layer are stored in a 2D
        # array with one row whereas the gradients of the
        # biases in the 3rd layer are stored in a 1D array.
        # That is why `approx` is a 1D array.
        for i in range(self.nb_hidden):
            pos = self.__parameters_eae['biases_decoder']['l1'].copy()
            pos[0, i] += offset
            hidden_decoder_pos = tls.leaky_relu(numpy.dot(y_tilde,
                self.__parameters_eae['weights_decoder']['l1']) +
                numpy.tile(pos, (nb_examples, 1)))
            reconstruction = numpy.dot(hidden_decoder_pos,
                self.__parameters_eae['weights_decoder']['mean']) + \
                numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                (nb_examples, 1))
            approx_pos = tls.loss_entropy_reconstruction(visible_units,
                                                         y_tilde,
                                                         reconstruction,
                                                         self.__parameters_fct,
                                                         self.nb_points_per_interval,
                                                         self.nb_intervals_per_side,
                                                         self.bin_width,
                                                         self.gamma)
            neg = self.__parameters_eae['biases_decoder']['l1'].copy()
            neg[0, i] -= offset
            hidden_decoder_neg = tls.leaky_relu(numpy.dot(y_tilde,
                self.__parameters_eae['weights_decoder']['l1']) +
                numpy.tile(neg, (nb_examples, 1)))
            reconstruction = numpy.dot(hidden_decoder_neg,
                self.__parameters_eae['weights_decoder']['mean']) + \
                numpy.tile(self.__parameters_eae['biases_decoder']['mean'],
                (nb_examples, 1))
            approx_neg = tls.loss_entropy_reconstruction(visible_units,
                                                         y_tilde,
                                                         reconstruction,
                                                         self.__parameters_fct,
                                                         self.nb_points_per_interval,
                                                         self.nb_intervals_per_side,
                                                         self.bin_width,
                                                         self.gamma)
            approx[i] = 0.5*(approx_pos - approx_neg)/offset
            if numpy.any(numpy.sign(hidden_decoder_pos) != numpy.sign(hidden_decoder_neg)):
                is_non_diff_lrelu = True
        
        if is_non_diff_lrelu:
            warnings.warn('Leaky ReLU wrecks the gradient checking of the 3rd set of biases. Re-run gradient checking.')
        else:
            tls.histogram(gbd_l1 - approx,
                          'Gradient checking biases (3)',
                          os.path.join(path_to_checking_g, 'gb_3.png'))
    
    def __checking_gw_2(self, visible_units, hidden_encoder, eps, gwe_latent, path_to_checking_g):
        """Runs gradient checking for the 2nd set of weights.
        
        The 2nd set of weights gathers the weights
        connecting the encoder hidden units to the
        latent variables.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        hidden_encoder : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Encoder hidden units.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Noise from the uniform distribution of support
            [`-0.5*self.bin_width`, `0.5*self.bin_width`].
        gwe_latent : numpy.ndarray
            2D array with data-type `numpy.float64`.
            2nd set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.nb_hidden, self.nb_y))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        
        # `idx_initial` stores the linear piece index
        # of each latent variable perturbed by uniform
        # noise before gradient checking.
        y_tilde_initial = numpy.dot(hidden_encoder,
            self.__parameters_eae['weights_encoder']['latent']) + \
            numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
            (nb_examples, 1)) + eps
        idx_initial = tls.index_linear_piece(y_tilde_initial.flatten(),
                                             self.nb_points_per_interval,
                                             self.nb_intervals_per_side)
        
        # `is_non_diff_fct` becomes true if the
        # non-differentiability of the piecewise
        # linear function at the edges of pieces
        # wrecks gradient checking.
        is_non_diff_lrelu = False
        is_non_diff_fct = False
        for i in range(self.nb_hidden):
            for j in range(self.nb_y):
                pos = self.__parameters_eae['weights_encoder']['latent'].copy()
                pos[i, j] += offset
                y_tilde = numpy.dot(hidden_encoder, pos) + \
                    numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
                    (nb_examples, 1)) + eps
                
                # `idx_pos` stores the linear piece index
                # of each latent variable perturbed by uniform
                # noise after adding an offset.
                idx_pos = tls.index_linear_piece(y_tilde.flatten(),
                                                 self.nb_points_per_interval,
                                                 self.nb_intervals_per_side)
                (hidden_decoder_pos, reconstruction) = self.decoder(y_tilde)
                loss_pos = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_pos = loss_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters_eae['weights_encoder']['latent'].copy()
                neg[i, j] -= offset
                y_tilde = numpy.dot(hidden_encoder, neg) + \
                    numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
                    (nb_examples, 1)) + eps
                
                # `idx_neg` stores the linear piece index
                # of each latent variable perturbed by uniform
                # noise after subtracting an offset.
                idx_neg = tls.index_linear_piece(y_tilde.flatten(),
                                                 self.nb_points_per_interval,
                                                 self.nb_intervals_per_side)
                (hidden_decoder_neg, reconstruction) = self.decoder(y_tilde)
                loss_neg = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_neg = loss_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                if numpy.any(numpy.sign(hidden_decoder_pos) != numpy.sign(hidden_decoder_neg)):
                    is_non_diff_lrelu = True
                
                # If the linear piece index of a latent
                # variable perturbed by uniform noise changes,
                # the non-differentiability of the piecewise
                # linear function at the piece edges kicks in.
                is_idx_pos_changed = not numpy.array_equal(idx_initial, idx_pos)
                is_idx_neg_changed = not numpy.array_equal(idx_initial, idx_neg)
                if is_idx_pos_changed or is_idx_neg_changed:
                    is_non_diff_fct = True
        
        if is_non_diff_lrelu:
            warnings.warn('Leaky ReLU wrecks the gradient checking of the 2nd set of weights. Re-run gradient checking.')
        elif is_non_diff_fct:
            warnings.warn('The piecewise linear function wrecks the gradient checking of the 2nd set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwe_latent - approx).flatten(),
                          'Gradient checking weights (2)',
                          os.path.join(path_to_checking_g, 'gw_2.png'))
    
    def __checking_gw_1(self, visible_units, hidden_encoder, eps, gwe_l1, path_to_checking_g):
        """Runs gradient checking for the 1st set of weights.
        
        The 1st set of weights gathers the weights
        connecting the visible units to the encoder
        hidden units.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        hidden_encoder : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Encoder hidden units.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Noise from the uniform distribution of support
            [`-0.5*self.bin_width`, `0.5*self.bin_width`].
        gwe_l1 : numpy.ndarray
            2D array with data-type `numpy.float64`.
            1st set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.nb_visible, self.nb_hidden))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        
        # `hidden_encoder` is only used for tracking
        # the errors due to the non-differentiability
        # of the piecewise linear function at the edges
        # of pieces.
        y_tilde_initial = numpy.dot(hidden_encoder,
            self.__parameters_eae['weights_encoder']['latent']) + \
            numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
            (nb_examples, 1)) + eps
        idx_initial = tls.index_linear_piece(y_tilde_initial.flatten(),
                                             self.nb_points_per_interval,
                                             self.nb_intervals_per_side)
        is_non_diff_lrelu = False
        is_non_diff_fct = False
        for i in range(self.nb_visible):
            for j in range(self.nb_hidden):
                pos = self.__parameters_eae['weights_encoder']['l1'].copy()
                pos[i, j] += offset
                hidden_encoder_pos = tls.leaky_relu(numpy.dot(visible_units, pos) +
                    numpy.tile(self.__parameters_eae['biases_encoder']['l1'],
                    (nb_examples, 1)))
                y_tilde = numpy.dot(hidden_encoder_pos,
                    self.__parameters_eae['weights_encoder']['latent']) + \
                    numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
                    (nb_examples, 1)) + eps
                idx_pos = tls.index_linear_piece(y_tilde.flatten(),
                                                 self.nb_points_per_interval,
                                                 self.nb_intervals_per_side)
                (hidden_decoder_pos, reconstruction) = self.decoder(y_tilde)
                loss_pos = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_pos = loss_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters_eae['weights_encoder']['l1'].copy()
                neg[i, j] -= offset
                hidden_encoder_neg = tls.leaky_relu(numpy.dot(visible_units, neg) +
                    numpy.tile(self.__parameters_eae['biases_encoder']['l1'],
                    (nb_examples, 1)))
                y_tilde = numpy.dot(hidden_encoder_neg,
                    self.__parameters_eae['weights_encoder']['latent']) + \
                    numpy.tile(self.__parameters_eae['biases_encoder']['latent'],
                    (nb_examples, 1)) + eps
                idx_neg = tls.index_linear_piece(y_tilde.flatten(),
                                                 self.nb_points_per_interval,
                                                 self.nb_intervals_per_side)
                (hidden_decoder_neg, reconstruction) = self.decoder(y_tilde)
                loss_neg = tls.loss_entropy_reconstruction(visible_units,
                                                           y_tilde,
                                                           reconstruction,
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side,
                                                           self.bin_width,
                                                           self.gamma)
                approx_neg = loss_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                is_0 = numpy.any(numpy.sign(hidden_encoder_pos) != numpy.sign(hidden_encoder_neg))
                is_1 = numpy.any(numpy.sign(hidden_decoder_pos) != numpy.sign(hidden_decoder_neg))
                if is_0 or is_1:
                    is_non_diff_lrelu = True
                is_idx_pos_changed = not numpy.array_equal(idx_initial, idx_pos)
                is_idx_neg_changed = not numpy.array_equal(idx_initial, idx_neg)
                if is_idx_pos_changed or is_idx_neg_changed:
                    is_non_diff_fct = True
        
        if is_non_diff_lrelu:
            warnings.warn('Leaky ReLU wrecks the gradient checking of the 1st set of weights. Re-run gradient checking.')
        elif is_non_diff_fct:
            warnings.warn('The piecewise linear function wrecks the gradient checking of the 1st set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwe_l1 - approx).flatten(),
                          'Gradient checking weights (1)',
                          os.path.join(path_to_checking_g, 'gw_1.png'))
    
    def __checking_gbw(self, visible_units, y, standard_eps, gradient_bw):
        """Runs gradient checking for the quantization bin width.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        y : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables.
        standard_eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Noise from the uniform distribution of
            support [-0.5, 0.5].
        gradient_bw : float
            Gradient of the quantization bin width.
        
        Raises
        ------
        AssertionError
            If the gradient checking of the quantization
            bin width fails.
        
        """
        offset = 1.e-4
        pos = self.bin_width + offset
        y_tilde = y + pos*standard_eps
        reconstruction = self.decoder(y_tilde)[1]
        approx_pos = tls.loss_entropy_reconstruction(visible_units,
                                                     y_tilde,
                                                     reconstruction,
                                                     self.__parameters_fct,
                                                     self.nb_points_per_interval,
                                                     self.nb_intervals_per_side,
                                                     pos,
                                                     self.gamma)
        neg = self.bin_width - offset
        y_tilde = y + neg*standard_eps
        reconstruction = self.decoder(y_tilde)[1]
        approx_neg = tls.loss_entropy_reconstruction(visible_units,
                                                     y_tilde,
                                                     reconstruction,
                                                     self.__parameters_fct,
                                                     self.nb_points_per_interval,
                                                     self.nb_intervals_per_side,
                                                     neg,
                                                     self.gamma)
        diff = gradient_bw - 0.5*((approx_pos - approx_neg).item())/offset
        assert abs(diff) < 1.e-8, \
            'The gradient checking of the quantization bin width fails.'
    
    def __checking_grid(self, max_abs_y):
        """Expands the grid and the parameters of the piecewise linear function if the condition of expansion is met.
        
        The condition of expansion is met when the
        largest absolute latent variable plus half
        the quantization bin width is larger than
        the number of unit intervals in the right
        half of the grid.
        
        Parameters
        ----------
        numpy.float64
            Largest absolute latent variable.
        
        """
        if max_abs_y + 0.5*self.bin_width >= self.nb_intervals_per_side:
            
            # In the case below, the size of the output from the
            # function `numpy.ceil` is equal to 1. The method
            # `numpy.ndarray.item` converts this output to a float.
            # `type(nb_added_per_side)` has to be equal to `int`.
            # If the above condition is an equality, `nb_added_per_side`
            # has to be strictly positive. That is why 1 is added.
            nb_added_per_side = int(numpy.ceil(max_abs_y + 0.5*self.bin_width).item()) - \
                self.nb_intervals_per_side + 1
            self.nb_intervals_per_side += nb_added_per_side
            self.__grid = numpy.linspace(-self.nb_intervals_per_side,
                                         self.nb_intervals_per_side,
                                         num=2*self.nb_points_per_interval*self.nb_intervals_per_side + 1)
            self.__parameters_fct = tls.expand_parameters(self.__parameters_fct,
                                                          self.__low_projection,
                                                          self.nb_points_per_interval,
                                                          nb_added_per_side)
    
    def backpropagation_fct(self, visible_units, is_checking=False, path_to_checking_g=''):
        """Computes the gradients of the parameters of the piecewise linear function.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        is_checking : bool, optional
            Is it the gradient checking mode? The default
            value is False.
        path_to_checking_g : str, optional
            Path to the folder storing the visualization
            of gradient checking. The default value is ''.
            
        Returns
        -------
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Gradients of the parameters of the piecewise
            linear function.
        
        """
        nb_examples = visible_units.shape[0]
        y = self.encoder(visible_units)[1]
        y_tilde = y + self.bin_width*tls.noise(nb_examples, self.nb_y)
        
        # When feeding `visible_units` into the
        # entropy autoencoder, is the condition
        # of expansion met?
        self.__checking_grid(numpy.amax(numpy.absolute(y)))
        gradients_fct = tls.gradient_density_approximation(y_tilde.flatten(),
                                                           self.__parameters_fct,
                                                           self.nb_points_per_interval,
                                                           self.nb_intervals_per_side)
        if is_checking and path_to_checking_g:
            self.__checking_gfct(y_tilde.flatten(),
                                 gradients_fct,
                                 path_to_checking_g)
        return gradients_fct
    
    def backpropagation_eae_bw(self, visible_units, is_checking=False, path_to_checking_g=''):
        """Computes the gradients of the parameters of the entropy autoencoder and that of the quantization bin width.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        is_checking : bool, optional
            Is it the gradient checking mode? The
            default value is False.
        path_to_checking_g : str, optional
            Path to the folder storing the visualization
            of gradient checking. The default value is ''.
            
        Returns
        -------
        dict
            Gradients of the parameters of the
            entropy autoencoder.
        float
            Gradient of the quantization bin width.
        
        """
        nb_examples = visible_units.shape[0]
        standard_eps = tls.noise(nb_examples, self.nb_y)
        eps = self.bin_width*standard_eps
        (hidden_encoder, y, y_tilde, hidden_decoder, reconstruction) = \
            self.forward_pass(visible_units, eps)
        
        # In the equations of backpropagation, the input to
        # `tls.leaky_relu_derivative` is the input to
        # Leaky ReLU in the decoder. But the activations
        # of Leaky ReLU are inserted into
        # `tls.leaky_relu_derivative`. In fact, the two
        # choices amount to the same as the function
        # `tls.leaky_relu_derivative` only considers
        # the sign of its input.
        delta_4 = reconstruction - visible_units
        delta_3 = numpy.dot(delta_4,
            numpy.transpose(self.__parameters_eae['weights_decoder']['mean']))*tls.leaky_relu_derivative(hidden_decoder)
        delta_2 = numpy.dot(delta_3,
            numpy.transpose(self.__parameters_eae['weights_decoder']['l1'])) + \
            self.gamma*tls.gradient_entropy(y_tilde,
                                            self.__parameters_fct,
                                            self.nb_points_per_interval,
                                            self.nb_intervals_per_side)
        delta_1 = numpy.dot(delta_2,
            numpy.transpose(self.__parameters_eae['weights_encoder']['latent']))*tls.leaky_relu_derivative(hidden_encoder)
        
        # Note that the entropy autoencoder parameters
        # stored in the dictionary named `self.__parameters_eae`
        # and their corresponding gradients stored in the
        # dictionary named `gradients_eae` are laid out in a
        # similar manner.
        gradients_eae = dict()
        
        # In Python 2.x, the standard division between two
        # integers returns an integer whereas, in Python 3.x,
        # the standard division between two integers returns
        # a float.
        gradients_eae['weights_encoder'] = {
            'l1': (1./nb_examples)*numpy.dot(numpy.transpose(visible_units), delta_1) + \
                self.__weights_decay_p*self.__parameters_eae['weights_encoder']['l1'],
            'latent': (1./nb_examples)*numpy.dot(numpy.transpose(hidden_encoder), delta_2) + \
                self.__weights_decay_p*self.__parameters_eae['weights_encoder']['latent']
        }
        gradients_eae['biases_encoder'] = {
            'l1': (1./nb_examples)*numpy.sum(delta_1, axis=0),
            'latent': (1./nb_examples)*numpy.sum(delta_2, axis=0)
        }
        gradients_eae['weights_decoder'] = {
            'l1': (1./nb_examples)*numpy.dot(numpy.transpose(y_tilde), delta_3) + \
                self.__weights_decay_p*self.__parameters_eae['weights_decoder']['l1'],
            'mean': (1./nb_examples)*numpy.dot(numpy.transpose(hidden_decoder), delta_4) + \
                self.__weights_decay_p*self.__parameters_eae['weights_decoder']['mean']
        }
        gradients_eae['biases_decoder'] = {
            'l1': (1./nb_examples)*numpy.sum(delta_3, axis=0),
            'mean': (1./nb_examples)*numpy.sum(delta_4, axis=0)
        }
        
        # `type(gradient_bw)` has to be equal to `float`
        # as `type(self.bin_width)` is equal to `float`.
        gradient_bw = (numpy.mean(numpy.sum(delta_2*standard_eps, axis=1)) -
            self.gamma/(numpy.log(2.)*self.bin_width)).item()
        
        # The bias gradients are less prone to errors
        # of derivation and errors of implementation
        # than the weight gradients.
        if is_checking and path_to_checking_g:
            self.__checking_gw_4(visible_units,
                                 y_tilde,
                                 hidden_decoder,
                                 gradients_eae['weights_decoder']['mean'],
                                 path_to_checking_g)
            self.__checking_gw_3(visible_units,
                                 y_tilde,
                                 gradients_eae['weights_decoder']['l1'],
                                 path_to_checking_g)
            self.__checking_gb_3(visible_units,
                                 y_tilde,
                                 gradients_eae['biases_decoder']['l1'],
                                 path_to_checking_g)
            self.__checking_gw_2(visible_units,
                                 hidden_encoder,
                                 eps,
                                 gradients_eae['weights_encoder']['latent'],
                                 path_to_checking_g)
            self.__checking_gw_1(visible_units,
                                 hidden_encoder,
                                 eps,
                                 gradients_eae['weights_encoder']['l1'],
                                 path_to_checking_g)
            self.__checking_gbw(visible_units,
                                y,
                                standard_eps,
                                gradient_bw)
        return (gradients_eae, gradient_bw)
    
    def training_fct(self, visible_units):
        """Trains the parameters of the piecewise linear function.
        
        The optimizer is stochastic gradient descent.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
            
        """
        gradients_fct = self.backpropagation_fct(visible_units)
        self.__parameters_fct -= self.__lr_fct*gradients_fct
        self.__parameters_fct = numpy.maximum(self.__parameters_fct,
                                              self.__low_projection)
    
    def __solver_eae(self, gradients_eae):
        """Updates the parameters of the entropy autoencoder.
        
        Parameters
        ----------
        gradients_eae : dict
            Gradients of the parameters of the entropy autoencoder.
        
        """
        for key_1 in gradients_eae.keys():
            for key_2 in gradients_eae[key_1].keys():
                self.__updates_eae[key_1][key_2] = \
                    self.__momentum_eae*self.__updates_eae[key_1][key_2] - \
                        self.__lr_eae*gradients_eae[key_1][key_2]
                self.__parameters_eae[key_1][key_2] += self.__updates_eae[key_1][key_2]
    
    def __solver_bw(self, gradient_bw):
        """Updates the quantization bin width.
        
        Parameters
        ----------
        gradient_bw : float
            Gradient of the quantization bin width.
        
        """
        self.bin_width -= self.__lr_bw*gradient_bw
        self.bin_width = max(self.bin_width, 0.1)
    
    def training_eae_bw(self, visible_units):
        """Trains the parameters of the entropy autoencoder and the quantization bin width if required.
        
        For the parameters of the entropy autoencoder,
        the optimizer is stochastic gradient descent
        with momentum. For the quantization bin width,
        the optimizer is stochastic gradient descent.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
            
        """
        (gradients_eae, gradient_bw) = self.backpropagation_eae_bw(visible_units)
        self.__solver_eae(gradients_eae)
        if self.is_bin_width_learned:
            self.__solver_bw(gradient_bw)
    
    def evaluation(self, visible_units):
        """Computes 6 indicators to assess how the training advances.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        
        Returns
        -------
        tuple
            numpy.float64
                Approximate entropy of the quantized latent variables.
            numpy.float64
                Entropy of the quantized latent variables.
            numpy.float64
                Scaled approximate entropy of the quantized
                latent variables.
            numpy.float64
                Error between the visible units and their
                reconstruction.
            numpy.float64
                Loss of the approximation of the probability
                density function of the latent variables perturbed
                by uniform noise with the piecewise linear function.
            int
                Number of dead quantized latent variables.
        
        """
        eps = self.bin_width*tls.noise(visible_units.shape[0], self.nb_y)
        (_, y, y_tilde, _, reconstruction) = self.forward_pass(visible_units, eps)
        
        # When feeding `visible_units` into the
        # entropy autoencoder, is the condition
        # of expansion met?
        self.__checking_grid(numpy.amax(numpy.absolute(y)))
        
        # For the function `tls.approximate_entropy`
        # and the function `tls.loss_density_approximation`,
        # the 1st argument is a 1D array.
        flattened_y_tilde = y_tilde.flatten()
        quantized_y = tls.quantization(y, self.bin_width)
        approx_entropy = tls.approximate_entropy(flattened_y_tilde,
                                                 self.__parameters_fct,
                                                 self.nb_points_per_interval,
                                                 self.nb_intervals_per_side,
                                                 self.bin_width)
        
        # In the function `tls.discrete_entropy`, `quantized_y`
        # is flattened to compute the entropy.
        disc_entropy = tls.discrete_entropy(quantized_y,
                                            self.bin_width)
        scaled_approx_entropy = self.gamma*approx_entropy
        rec_error = tls.reconstruction_error(visible_units,
                                             reconstruction,
                                             True)
        loss_density_approx = tls.loss_density_approximation(flattened_y_tilde,
                                                             self.__parameters_fct,
                                                             self.nb_points_per_interval,
                                                             self.nb_intervals_per_side)
        nb_dead = tls.count_zero_columns(quantized_y)
        return (approx_entropy,
                disc_entropy,
                scaled_approx_entropy,
                rec_error,
                loss_density_approx,
                nb_dead)
    
    def weights_decay(self):
        """Computes the l2-norm weight decay.
        
        Returns
        -------
        numpy.float64
            L2-norm weight decay.
        
        """
        accumulation = 0.
        for key_1 in ['weights_encoder', 'weights_decoder']:
            for key_2 in self.__parameters_eae[key_1].keys():
                accumulation += numpy.sum(self.__parameters_eae[key_1][key_2]**2)
        return 0.5*self.__weights_decay_p*accumulation
    
    def area_under_piecewise_linear_function(self):
        """Computes the area under the piecewise linear function.
        
        Returns
        -------
        numpy.float64
            Area under the piecewise linear function.
        
        """
        return tls.area_under_piecewise_linear_function(self.__parameters_fct,
                                                        self.nb_points_per_interval)
    
    def checking_activations(self, visible_units, title_hist_0, title_hist_1, title_hist_2,
                             path_hist_0, path_hist_1, path_hist_2, path_image):
        """Creates visualizations of the encoder activations and saves them.
        
        Note that all visualizations are
        gathered in a single method to avoid
        running many times the same forward pass.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        title_hist_0 : str
            Title of the 1st histogram.
        title_hist_1 : str
            Title of the 2nd histogram.
        title_hist_2 : str
            Title of the 3rd histogram.
        path_hist_0 : str
            Path to the 1st saved histogram. The
            path must end with ".png".
        path_hist_1 : str
            Path to the 2nd saved histogram. The
            path must end with ".png".
        path_hist_2 : str
            Path to the 3rd saved histogram. The
            path must end with ".png".
        path_image : str
            Path to the saved image. The path
            must end with ".png".
        
        """
        (hidden_encoder, y) = self.encoder(visible_units)
        y_tilde = y + self.bin_width*tls.noise(visible_units.shape[0], self.nb_y)
        quantized_y = tls.quantization(y, self.bin_width)
        tls.histogram(hidden_encoder.flatten(),
                      title_hist_0,
                      path_hist_0)
        tls.histogram(y.flatten(),
                      title_hist_1,
                      path_hist_1)
        tls.normed_histogram(y_tilde,
                             self.__grid,
                             self.__parameters_fct,
                             title_hist_2,
                             path_hist_2)
        tls.visualize_dead(quantized_y, path_image)
    
    def checking_p_1(self, key_type, key_layer):
        """Divides the parameter updates mean magnitude by the parameters mean magnitude.
        
        A single set of weights/biases is considered.
        To find this set in the dictionary that stores
        the parameters of the entropy autoencoder,
        two keys are required.
        
        Parameters
        ----------
        key_type : str
            1st key. It is equal to either
            "weights_encoder" or "biases_encoder" if
            this set is in the encoder. It is equal
            to either "weights_decoder" or "biases_decoder"
            if this set is in the decoder.
        key_layer : str
            2nd key. It is equal to either "l1" or "mean".
        
        Returns
        -------
        numpy.float64
            Parameter updates mean magnitude divided
            by the parameters mean magnitude.
        
        """
        parameters = self.__parameters_eae[key_type][key_layer].flatten()
        updates = self.__updates_eae[key_type][key_layer].flatten()
        return numpy.mean(numpy.absolute(updates))/numpy.mean(numpy.absolute(parameters))
    
    def checking_p_2(self, key_type, key_layer, title, title_update, path, path_update):
        """Creates a histogram of parameters, a histogram of their update and saves the two histograms.
        
        A single set of weights/biases is considered.
        To find this set in the dictionary that stores
        the parameters of the entropy autoencoder,
        two keys are required.
        
        Parameters
        ----------
        key_type : str
            1st key. It is equal to either
            "weights_encoder" or "biases_encoder" if
            this set is in the encoder. It is equal
            to either "weights_decoder" or "biases_decoder"
            if this set is in the decoder.
        key_layer : str
            2nd key. It is equal to either "l1" or "mean".
        title : str
            Title of the histogram of the parameters.
        title_update : str
            Title of the histogram of the parameter updates.
        path : str
            Path to the saved histogram of the parameters.
            The path must end with ".png".
        path_update : str
            Path to the saved histogram of the parameter
            updates. The path must end with ".png".
        
        """
        tls.histogram(self.__parameters_eae[key_type][key_layer].flatten(),
                      title,
                      path)
        tls.histogram(self.__updates_eae[key_type][key_layer].flatten(),
                      title_update,
                      path_update)
    
    def checking_p_3(self, is_encoder, nb_display, height_image, width_image, nb_vertically, path):
        """Arranges the weight filters in a single RGB image and saves the single image.
        
        Parameters
        ----------
        is_encoder : bool
            If True, the weight filters in the
            1st encoder layer are considered. Otherwise,
            the weight filters in the 2nd decoder layer
            are considered.
        nb_display : int
            Number of weight filters in the single
            RGB image.
        height_image : int
            Image height.
        width_image : int
            Image width.
        nb_vertically : int
            Number of weight filters per column
            in the single RGB image.
        path : str
            Path to the saved single image. The
            path must end with ".png".
        
        """
        if is_encoder:
            weights = numpy.transpose(self.__parameters_eae['weights_encoder']['l1'][:, 0:nb_display])
        else:
            weights = self.__parameters_eae['weights_decoder']['mean'][0:nb_display, :]
        tls.visualize_weights(weights,
                              height_image,
                              width_image,
                              nb_vertically,
                              path)


