"""A library that defines the class `VariationalAutoencoder`."""

import numpy
import os
import warnings

import tools.tools as tls


class VariationalAutoencoder(object):
    """Variational autoencoder class.
    
    All of its attributes are conventionally private.
    
    """
    
    def __init__(self, nb_visible, nb_hidden, nb_z, is_continuous, alpha,
                 learning_rate=1.e-4, momentum=0.9, weights_decay_p=5.e-4):
        """Initializes the variational autoencoder.
        
        This initialization includes the initialization
        of the parameters of the variational autoencoder,
        i.e weights and biases, and the initialization of
        the training hyperparameters.
        
        Parameters
        ----------
        nb_visible : int
            Number of visible units.
        nb_hidden : int
            Number of recognition hidden units.
            The number of generation hidden units
            is equal to the number of recognition
            hidden units.
        nb_z : int
            Number of latent variables.
        is_continuous : bool
            Is each visible unit modeled as a
            continuous random variable with Gaussian
            probability density function?
        alpha : float
            Scaling coefficient.
        learning_rate : float, optional
            Learning rate. The default value is 1.0e-4.
        momentum : float, optional
            Momentum. The default value is 0.9.
        weights_decay_p : float, optional
            Coefficient that weights the l2-norm weight
            decay with respect to the opposite of the
            variational lower bound in the objective
            function. The default value is 5.0e-4.
        
        """
        self.__nb_visible = nb_visible
        self.__nb_hidden = nb_hidden
        self.__nb_z = nb_z
        self.__is_continuous = is_continuous
        self.__alpha = alpha
        self.__parameters = self.__initialize_parameters()
        self.__updates = self.__initialize_updates()
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__weights_decay_p = weights_decay_p
    
    @property
    def nb_z(self):
        """Returns the number of latent variables."""
        return self.__nb_z
    
    @property
    def learning_rate(self):
        """Returns a tuple containing two learning rates.
        
        The 1st learning rate is used to train
        the weights and the biases in the mean
        layer of the recognition network. The 2nd
        learning rate is used to train all the
        other weights and biases.
        
        """
        return (0.1*self.__learning_rate, self.__learning_rate)

    def __initialize_parameters(self):
        """Initializes the parameters of the variational autoencoder, i.e weights and biases.
        
        Returns
        -------
        dict
            Parameters of the variational autoencoder.
        
        """
        parameters = dict()
        parameters['weights_recognition'] = {
            'l1': numpy.random.normal(loc=0.,
                                      scale=0.01,
                                      size=(self.__nb_visible, self.__nb_hidden)),
            'mean': numpy.random.normal(loc=0.,
                                        scale=0.01,
                                        size=(self.__nb_hidden, self.__nb_z)),
            'log_std_squared': numpy.random.normal(loc=0.,
                                                   scale=0.01,
                                                   size=(self.__nb_hidden, self.__nb_z))
        }
        parameters['biases_recognition'] = {
            'l1': numpy.zeros((1, self.__nb_hidden)),
            'mean': numpy.zeros((1, self.__nb_z)),
            'log_std_squared': numpy.zeros((1, self.__nb_z))
        }
        parameters['weights_generation'] = {
            'l1': numpy.random.normal(loc=0.,
                                      scale=0.01,
                                      size=(self.__nb_z, self.__nb_hidden)),
            'mean': numpy.random.normal(loc=0.,
                                        scale=0.01,
                                        size=(self.__nb_hidden, self.__nb_visible))
        }
        parameters['biases_generation'] = {
            'l1': numpy.zeros((1, self.__nb_hidden)),
            'mean': numpy.zeros((1, self.__nb_visible))
        }
        return parameters
    
    def __initialize_updates(self):
        """Initializes the updates of the parameters of the variational autoencoder.
        
        Returns
        -------
        dict
            Updates of the parameters of the
            variational autoencoder.
        
        """
        updates = dict()
        updates['weights_recognition'] = {
            'l1': numpy.zeros((self.__nb_visible, self.__nb_hidden)),
            'mean': numpy.zeros((self.__nb_hidden, self.__nb_z)),
            'log_std_squared': numpy.zeros((self.__nb_hidden, self.__nb_z))
        }
        updates['biases_recognition'] = {
            'l1': numpy.zeros((1, self.__nb_hidden)),
            'mean': numpy.zeros((1, self.__nb_z)),
            'log_std_squared': numpy.zeros((1, self.__nb_z))
        }
        updates['weights_generation'] = {
            'l1': numpy.zeros((self.__nb_z, self.__nb_hidden)),
            'mean': numpy.zeros((self.__nb_hidden, self.__nb_visible))
        }
        updates['biases_generation'] = {
            'l1': numpy.zeros((1, self.__nb_hidden)),
            'mean': numpy.zeros((1, self.__nb_visible))
        }
        
        # Note that the variational autoencoder parameters
        # stored in the dictionary named `self.__parameters`
        # and their corresponding updates stored in the dictionary
        # named `updates` are laid out in a similar manner.
        return updates
    
    def recognition_network(self, visible_units):
        """Computes the neural activations in the recognition network.
        
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
                Recognition hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Mean of each latent variable normal
                distribution.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Log standard deviation squared of each
                latent variable normal distribution.
        
        """
        nb_examples = visible_units.shape[0]
        hidden_recognition = tls.relu(numpy.dot(visible_units,
            self.__parameters['weights_recognition']['l1']) +
            numpy.tile(self.__parameters['biases_recognition']['l1'],
            (nb_examples, 1)))
        z_mean = numpy.dot(hidden_recognition,
            self.__parameters['weights_recognition']['mean']) + \
            numpy.tile(self.__parameters['biases_recognition']['mean'],
            (nb_examples, 1))
        z_log_std_squared = numpy.dot(hidden_recognition,
            self.__parameters['weights_recognition']['log_std_squared']) + \
            numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
            (nb_examples, 1))
        return (hidden_recognition, z_mean, z_log_std_squared)
    
    def generation_network(self, z):
        """Computes the neural activations in the generation network.
        
        Parameters
        ----------
        z : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables.
            
        Returns
        -------
        tuple
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Generation hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Reconstruction of the visible units.
        
        """
        nb_examples = z.shape[0]
        hidden_generation = tls.relu(numpy.dot(z,
            self.__parameters['weights_generation']['l1']) +
            numpy.tile(self.__parameters['biases_generation']['l1'],
            (nb_examples, 1)))
        if self.__is_continuous:
            reconstruction = numpy.dot(hidden_generation,
                self.__parameters['weights_generation']['mean']) + \
                numpy.tile(self.__parameters['biases_generation']['mean'],
                (nb_examples, 1))
        else:
            reconstruction = tls.sigmoid(numpy.dot(hidden_generation,
                self.__parameters['weights_generation']['mean']) +
                numpy.tile(self.__parameters['biases_generation']['mean'],
                (nb_examples, 1)))
        return (hidden_generation, reconstruction)
    
    def forward_pass(self, visible_units, epsilon=None):
        """Computes the neural activations in the variational autoencoder.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        epsilon : numpy.ndarray, optional
            2D array with data-type `numpy.float64`.
            Samples from the standard normal distribution.
            The default value is None.
        
        Returns
        -------
        tuple
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Recognition hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Mean of each latent variable normal
                distribution.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Log standard deviation squared of each
                latent variable normal distribution.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Latent variables.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Generation hidden units.
            numpy.ndarray
                2D array with data-type `numpy.float64`.
                Reconstruction of the visible units.
        
        """
        (hidden_recognition, z_mean, z_log_std_squared) = self.recognition_network(visible_units)
        if epsilon is None:
            
            # The method `numpy.ndarray.copy` allocates
            # memory to store a copy of the array
            # whereas a simple equal sign creates
            # a reference to the array.
            z = z_mean.copy()
        else:
            z = z_mean + numpy.exp(0.5*z_log_std_squared)*epsilon
        (hidden_generation, reconstruction) = self.generation_network(z)
        return (hidden_recognition, z_mean, z_log_std_squared, z, hidden_generation, reconstruction)
    
    def __checking_gw_5(self, visible_units, z_mean, z_log_std_squared,
                        hidden_generation, gwg_mean, path_to_checking_g):
        """Runs gradient checking for the 5th set of weights.
        
        The 5th set of weights gathers the weights
        connecting the generation hidden units to the
        reconstruction of the visible units.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        z_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Mean of each latent variable normal
            distribution.
        z_log_std_squared : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Log standard deviation squared of each
            latent variable normal distribution.
        hidden generation : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Generation hidden units.
        gwg_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            5th set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.__nb_hidden, self.__nb_visible))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        
        # The dictionary containing all the
        # parameters should not be modified
        # by the methods running gradient checking.
        # The l2-norm weight decay contains some terms
        # that do not cancel out during gradient checking.
        for i in range(self.__nb_hidden):
            for j in range(self.__nb_visible):
                pos = self.__parameters['weights_generation']['mean'].copy()
                pos[i, j] += offset
                if self.__is_continuous:
                    reconstruction = numpy.dot(hidden_generation, pos) + \
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1))
                else:
                    reconstruction = tls.sigmoid(numpy.dot(hidden_generation, pos) +
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1)))
                vlb_pos = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_pos = vlb_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters['weights_generation']['mean'].copy()
                neg[i, j] -= offset
                if self.__is_continuous:
                    reconstruction = numpy.dot(hidden_generation, neg) + \
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1))
                else:
                    reconstruction = tls.sigmoid(numpy.dot(hidden_generation, neg) +
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1)))
                vlb_neg = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_neg = vlb_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
        
        tls.histogram((gwg_mean - approx).flatten(),
                      'Gradient checking weights (5)',
                      os.path.join(path_to_checking_g, 'gw_5.png'))
    
    def __checking_gw_4(self, visible_units, z_mean, z_log_std_squared, z, gwg_l1, path_to_checking_g):
        """Runs gradient checking for the 4th set of weights.
        
        The 4th set of weights gathers the weights
        connecting the latent variables to the
        generation hidden units.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        z_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Mean of each latent variable normal
            distribution.
        z_log_std_squared : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Log standard deviation squared of each
            latent variable normal distribution.
        z : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Latent variables.
        gwg_l1 : numpy.ndarray
            2D array with data-type `numpy.float64`.
            4th set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.__nb_z, self.__nb_hidden))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        
        # `is_non_diff_relu` becomes true if the
        # non-differentiability of ReLU at 0 wrecks
        # gradient checking.
        is_non_diff_relu = False
        for i in range(self.__nb_z):
            for j in range(self.__nb_hidden):
                pos = self.__parameters['weights_generation']['l1'].copy()
                pos[i, j] += offset
                hidden_generation_pos = tls.relu(numpy.dot(z, pos) +
                    numpy.tile(self.__parameters['biases_generation']['l1'],
                    (nb_examples, 1)))
                if self.__is_continuous:
                    reconstruction = numpy.dot(hidden_generation_pos,
                        self.__parameters['weights_generation']['mean']) + \
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1))
                else:
                    reconstruction = tls.sigmoid(numpy.dot(hidden_generation_pos,
                        self.__parameters['weights_generation']['mean']) +
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1)))
                vlb_pos = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_pos = vlb_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters['weights_generation']['l1'].copy()
                neg[i, j] -= offset
                hidden_generation_neg = tls.relu(numpy.dot(z, neg) +
                    numpy.tile(self.__parameters['biases_generation']['l1'],
                    (nb_examples, 1)))
                if self.__is_continuous:
                    reconstruction = numpy.dot(hidden_generation_neg,
                        self.__parameters['weights_generation']['mean']) + \
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1))
                else:
                    reconstruction = tls.sigmoid(numpy.dot(hidden_generation_neg,
                        self.__parameters['weights_generation']['mean']) +
                        numpy.tile(self.__parameters['biases_generation']['mean'],
                        (nb_examples, 1)))
                vlb_neg = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_neg = vlb_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                if numpy.any(numpy.sign(hidden_generation_pos) != numpy.sign(hidden_generation_neg)):
                    is_non_diff_relu = True
        
        # If ReLU wrecks the gradient checking, the
        # difference between the gradient an its finite
        # difference approximation is not saved as it
        # is not interpretable.
        if is_non_diff_relu:
            warnings.warn('ReLU wrecks the gradient checking of the 4th set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwg_l1 - approx).flatten(),
                          'Gradient checking weights (4)',
                          os.path.join(path_to_checking_g, 'gw_4.png'))
    
    def __checking_gw_3(self, visible_units, hidden_recognition, z_mean, eps,
                        gwr_log_std_squared, path_to_checking_g):
        """Runs gradient checking for the 3rd set of weights.
        
        The 3rd set of weights gathers the weights
        connecting the recognition hidden units to
        the log standard deviation squared of each
        latent variable normal distribution.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        hidden_recognition : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Recognition hidden units.
        z_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Mean of each latent variable normal
            distribution.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Samples from the standard normal
            distribution.
        gwr_log_std_squared : numpy.ndarray
            2D array with data-type `numpy.float64`.
            3rd set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.__nb_hidden, self.__nb_z))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_relu = False
        for i in range(self.__nb_hidden):
            for j in range(self.__nb_z):
                pos = self.__parameters['weights_recognition']['log_std_squared'].copy()
                pos[i, j] += offset
                z_log_std_squared = numpy.dot(hidden_recognition, pos) + \
                    numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_pos, reconstruction) = self.generation_network(z)
                vlb_pos = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_pos = vlb_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters['weights_recognition']['log_std_squared'].copy()
                neg[i, j] -= offset
                z_log_std_squared = numpy.dot(hidden_recognition, neg) + \
                    numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_neg, reconstruction) = self.generation_network(z)
                vlb_neg = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_neg = vlb_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                if numpy.any(numpy.sign(hidden_generation_pos) != numpy.sign(hidden_generation_neg)):
                    is_non_diff_relu = True
        
        if is_non_diff_relu:
            warnings.warn('ReLU wrecks the gradient checking of the 3rd set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwr_log_std_squared - approx).flatten(),
                          'Gradient checking weights (3)',
                          os.path.join(path_to_checking_g, 'gw_3.png'))
    
    def __checking_gw_2(self, visible_units, hidden_recognition, z_log_std_squared,
                        eps, gwr_mean, path_to_checking_g):
        """Runs gradient checking for the 2nd set of weights.
        
        The 2nd set of weights gathers the
        weights connecting the recognition hidden
        units to the mean of each latent variable
        normal distribution.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        hidden_recognition : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Recognition hidden units.
        z_log_std_squared : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Log standard deviation squared of each
            latent variable normal distribution.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Samples from the standard normal
            distribution.
        gwr_mean : numpy.ndarray
            2D array with data-type `numpy.float64`.
            2nd set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.__nb_hidden, self.__nb_z))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_relu = False
        for i in range(self.__nb_hidden):
            for j in range(self.__nb_z):
                pos = self.__parameters['weights_recognition']['mean'].copy()
                pos[i, j] += offset
                z_mean = numpy.dot(hidden_recognition, pos) + \
                    numpy.tile(self.__parameters['biases_recognition']['mean'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_pos, reconstruction) = self.generation_network(z)
                vlb_pos = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_pos = vlb_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters['weights_recognition']['mean'].copy()
                neg[i, j] -= offset
                z_mean = numpy.dot(hidden_recognition, neg) + \
                    numpy.tile(self.__parameters['biases_recognition']['mean'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_neg, reconstruction) = self.generation_network(z)
                vlb_neg = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_neg = vlb_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                if numpy.any(numpy.sign(hidden_generation_pos) != numpy.sign(hidden_generation_neg)):
                    is_non_diff_relu = True
        
        if is_non_diff_relu:
            warnings.warn('ReLU wrecks the gradient checking of the 2nd set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwr_mean - approx).flatten(),
                          'Gradient checking weights (2)',
                          os.path.join(path_to_checking_g, 'gw_2.png'))
    
    def __checking_gw_1(self, visible_units, eps, gwr_l1, path_to_checking_g):
        """Runs gradient checking for the 1st set of weights.
        
        The 1st set of weights gathers the weights
        connecting the visible units to the recognition
        hidden units.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Samples from the standard normal
            distribution.
        gwr_l1 : numpy.ndarray
            2D array with data-type `numpy.float64`.
            1st set of weight gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros((self.__nb_visible, self.__nb_hidden))
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_relu = False
        for i in range(self.__nb_visible):
            for j in range(self.__nb_hidden):
                pos = self.__parameters['weights_recognition']['l1'].copy()
                pos[i, j] += offset
                hidden_recognition_pos = tls.relu(numpy.dot(visible_units, pos) +
                    numpy.tile(self.__parameters['biases_recognition']['l1'],
                    (nb_examples, 1)))
                z_mean = numpy.dot(hidden_recognition_pos,
                    self.__parameters['weights_recognition']['mean']) + \
                    numpy.tile(self.__parameters['biases_recognition']['mean'],
                    (nb_examples, 1))
                z_log_std_squared = numpy.dot(hidden_recognition_pos,
                    self.__parameters['weights_recognition']['log_std_squared']) + \
                    numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_pos, reconstruction) = self.generation_network(z)
                vlb_pos = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_pos = vlb_pos + 0.5*self.__weights_decay_p*numpy.sum(pos**2)
                neg = self.__parameters['weights_recognition']['l1'].copy()
                neg[i, j] -= offset
                hidden_recognition_neg = tls.relu(numpy.dot(visible_units, neg) +
                    numpy.tile(self.__parameters['biases_recognition']['l1'],
                    (nb_examples, 1)))
                z_mean = numpy.dot(hidden_recognition_neg,
                    self.__parameters['weights_recognition']['mean']) + \
                    numpy.tile(self.__parameters['biases_recognition']['mean'],
                    (nb_examples, 1))
                z_log_std_squared = numpy.dot(hidden_recognition_neg,
                    self.__parameters['weights_recognition']['log_std_squared']) + \
                    numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                    (nb_examples, 1))
                z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
                (hidden_generation_neg, reconstruction) = self.generation_network(z)
                vlb_neg = tls.opposite_vlb(visible_units,
                                           z_mean,
                                           z_log_std_squared,
                                           reconstruction,
                                           self.__alpha,
                                           self.__is_continuous)
                approx_neg = vlb_neg + 0.5*self.__weights_decay_p*numpy.sum(neg**2)
                approx[i, j] = 0.5*(approx_pos - approx_neg)/offset
                is_0 = numpy.any(numpy.sign(hidden_recognition_pos) != numpy.sign(hidden_recognition_neg))
                is_1 = numpy.any(numpy.sign(hidden_generation_pos) != numpy.sign(hidden_generation_neg))
                if is_0 or is_1:
                    is_non_diff_relu = True
        
        if is_non_diff_relu:
            warnings.warn('ReLU wrecks the gradient checking of the 1st set of weights. Re-run gradient checking.')
        else:
            tls.histogram((gwr_l1 - approx).flatten(),
                          'Gradient checking weights (1)',
                          os.path.join(path_to_checking_g, 'gw_1.png'))
    
    def __checking_gb_1(self, visible_units, eps, gbr_l1, path_to_checking_g):
        """Runs gradient checking for the 1st set of biases.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        eps : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Samples from the standard normal
            distribution.
        gbr_l1 : numpy.ndarray
            1D array with data-type `numpy.float64`.
            1st set of bias gradients.
        path_to_checking_g : str
            Path to the folder storing the visualization
            of gradient checking.
        
        """
        approx = numpy.zeros(self.__nb_hidden)
        offset = 1.e-4
        nb_examples = visible_units.shape[0]
        is_non_diff_relu = False
        
        # The biases in the 1st layer are stored in
        # a 2D array with one row whereas the gradients
        # of the biases in the 1st layer are stored in a
        # 1D array. That is why `approx` is a 1D array.
        for i in range(self.__nb_hidden):
            pos = self.__parameters['biases_recognition']['l1'].copy()
            pos[0, i] += offset
            hidden_recognition_pos = tls.relu(numpy.dot(visible_units,
                self.__parameters['weights_recognition']['l1']) +
                numpy.tile(pos, (nb_examples, 1)))
            z_mean = numpy.dot(hidden_recognition_pos,
                self.__parameters['weights_recognition']['mean']) + \
                numpy.tile(self.__parameters['biases_recognition']['mean'],
                (nb_examples, 1))
            z_log_std_squared = numpy.dot(hidden_recognition_pos,
                self.__parameters['weights_recognition']['log_std_squared']) + \
                numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                (nb_examples, 1))
            z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
            (hidden_generation_pos, reconstruction) = self.generation_network(z)
            approx_pos = tls.opposite_vlb(visible_units,
                                          z_mean,
                                          z_log_std_squared,
                                          reconstruction,
                                          self.__alpha,
                                          self.__is_continuous)
            neg = self.__parameters['biases_recognition']['l1'].copy()
            neg[0, i] -= offset
            hidden_recognition_neg = tls.relu(numpy.dot(visible_units,
                self.__parameters['weights_recognition']['l1']) +
                numpy.tile(neg, (nb_examples, 1)))
            z_mean = numpy.dot(hidden_recognition_neg,
                self.__parameters['weights_recognition']['mean']) + \
                numpy.tile(self.__parameters['biases_recognition']['mean'],
                (nb_examples, 1))
            z_log_std_squared = numpy.dot(hidden_recognition_neg,
                self.__parameters['weights_recognition']['log_std_squared']) + \
                numpy.tile(self.__parameters['biases_recognition']['log_std_squared'],
                (nb_examples, 1))
            z = z_mean + numpy.exp(0.5*z_log_std_squared)*eps
            (hidden_generation_neg, reconstruction) = self.generation_network(z)
            approx_neg = tls.opposite_vlb(visible_units,
                                          z_mean,
                                          z_log_std_squared,
                                          reconstruction,
                                          self.__alpha,
                                          self.__is_continuous)
            approx[i] = 0.5*(approx_pos - approx_neg)/offset
            is_0 = numpy.any(numpy.sign(hidden_recognition_pos) != numpy.sign(hidden_recognition_neg))
            is_1 = numpy.any(numpy.sign(hidden_generation_pos) != numpy.sign(hidden_generation_neg))
            if is_0 or is_1:
                is_non_diff_relu = True
        
        if is_non_diff_relu:
            warnings.warn('ReLU wrecks the gradient checking of the 1st set of biases. Re-run gradient checking.')
        else:
            tls.histogram(gbr_l1 - approx,
                          'Gradient checking biases (1)',
                          os.path.join(path_to_checking_g, 'gb_1.png'))
    
    def backpropagation(self, visible_units, is_checking=False, path_to_checking_g=''):
        """Computes the gradients of the parameters of the variational autoencoder.
        
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
            variational autoencoder.
        
        """
        nb_examples = visible_units.shape[0]
        eps = numpy.random.normal(loc=0., scale=1., size=(nb_examples, self.__nb_z))
        (hidden_recognition, z_mean, z_log_std_squared, z, hidden_generation, reconstruction) = \
            self.forward_pass(visible_units, epsilon=eps)
        
        # In the equations of backpropagation, the input to
        # `tls.relu_derivative` is the input to ReLU in
        # the generation network. But the activations of ReLU
        # are inserted into `tls.relu_derivative`. In fact,
        # the two choices amount to the same as the function
        # `tls.relu_derivative` only considers the sign of
        # its input.
        delta_5 = reconstruction - visible_units
        delta_4 = numpy.dot(delta_5,
            numpy.transpose(self.__parameters['weights_generation']['mean']))*tls.relu_derivative(hidden_generation)
        delta_3 = 0.5*(self.__alpha*(numpy.exp(z_log_std_squared) - 1.) +
            numpy.dot(delta_4,
            numpy.transpose(self.__parameters['weights_generation']['l1']))*numpy.exp(0.5*z_log_std_squared)*eps)
        delta_2 = self.__alpha*z_mean + numpy.dot(delta_4,
            numpy.transpose(self.__parameters['weights_generation']['l1']))
        delta_1 = (numpy.dot(delta_3,
            numpy.transpose(self.__parameters['weights_recognition']['log_std_squared'])) +
            numpy.dot(delta_2,
            numpy.transpose(self.__parameters['weights_recognition']['mean'])))*tls.relu_derivative(hidden_recognition)
        
        # Note that the variational autoencoder
        # parameters stored in the dictionary named
        # `self.__parameters` and their corresponding
        # gradients stored in the dictionary named
        # `gradients` are laid out in a similar manner.
        gradients = dict()
        
        # In Python 2.x, the standard division between
        # two integers returns an integer whereas,
        # in Python 3.x, the standard division between two
        # integers returns a float.
        gradients['weights_recognition'] = {
            'l1': (1./nb_examples)*numpy.dot(numpy.transpose(visible_units), delta_1) + \
                self.__weights_decay_p*self.__parameters['weights_recognition']['l1'],
            'mean': (1./nb_examples)*numpy.dot(numpy.transpose(hidden_recognition), delta_2) + \
                self.__weights_decay_p*self.__parameters['weights_recognition']['mean'],
            'log_std_squared': (1./nb_examples)*numpy.dot(numpy.transpose(hidden_recognition), delta_3) + \
                self.__weights_decay_p*self.__parameters['weights_recognition']['log_std_squared']
        }
        gradients['biases_recognition'] = {
            'l1': (1./nb_examples)*numpy.sum(delta_1, axis=0),
            'mean': (1./nb_examples)*numpy.sum(delta_2, axis=0),
            'log_std_squared': (1./nb_examples)*numpy.sum(delta_3, axis=0)
        }
        gradients['weights_generation'] = {
            'l1': (1./nb_examples)*numpy.dot(numpy.transpose(z), delta_4) + \
                self.__weights_decay_p*self.__parameters['weights_generation']['l1'],
            'mean': (1./nb_examples)*numpy.dot(numpy.transpose(hidden_generation), delta_5) + \
                self.__weights_decay_p*self.__parameters['weights_generation']['mean']
        }
        gradients['biases_generation'] = {
            'l1': (1./nb_examples)*numpy.sum(delta_4, axis=0),
            'mean': (1./nb_examples)*numpy.sum(delta_5, axis=0)
        }
        
        # The bias gradients are less prone to
        # errors of derivation and errors of
        # implementation than the weight gradients.
        if is_checking and path_to_checking_g:
            self.__checking_gw_5(visible_units,
                                 z_mean,
                                 z_log_std_squared,
                                 hidden_generation,
                                 gradients['weights_generation']['mean'],
                                 path_to_checking_g)
            self.__checking_gw_4(visible_units,
                                 z_mean,
                                 z_log_std_squared,
                                 z,
                                 gradients['weights_generation']['l1'],
                                 path_to_checking_g)
            self.__checking_gw_3(visible_units,
                                 hidden_recognition,
                                 z_mean,
                                 eps,
                                 gradients['weights_recognition']['log_std_squared'],
                                 path_to_checking_g)
            self.__checking_gw_2(visible_units,
                                 hidden_recognition,
                                 z_log_std_squared,
                                 eps,
                                 gradients['weights_recognition']['mean'],
                                 path_to_checking_g)
            self.__checking_gw_1(visible_units,
                                 eps,
                                 gradients['weights_recognition']['l1'],
                                 path_to_checking_g)
            self.__checking_gb_1(visible_units,
                                 eps,
                                 gradients['biases_recognition']['l1'],
                                 path_to_checking_g)
        return gradients
    
    def __solver(self, gradients):
        """Updates the parameters of the variational autoencoder.
        
        Parameters
        ----------
        gradients : dict
            Gradients of the parameters of the
            variational autoencoder.
        
        """
        for key_1 in gradients.keys():
            is_recognition = key_1 in ['weights_recognition', 'biases_recognition']
            for key_2 in gradients[key_1].keys():
                
                # In the 2nd layer of the recognition network,
                # the learning is relatively unstable.
                if is_recognition and key_2 == 'mean':
                    lr = 0.1*self.__learning_rate
                else:
                    lr = self.__learning_rate
                
                # Python knows how to handle operations
                # between a 2D array with one row and a
                # 1D array. The bias gradients do not
                # need to be reshaped from a 1D array
                # to a 2D array with one row.
                self.__updates[key_1][key_2] = \
                    self.__momentum*self.__updates[key_1][key_2] - lr*gradients[key_1][key_2]
                self.__parameters[key_1][key_2] += self.__updates[key_1][key_2]
    
    def training(self, visible_units):
        """Trains the parameters of the variational autoencoder.
        
        The optimizer is stochastic gradient descent
        with momentum.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
            
        """
        self.__solver(self.backpropagation(visible_units))
    
    def evaluation(self, visible_units):
        """Computes 2 indicators to assess how the training advances.
        
        Parameters
        ----------
        visible_units : numpy.ndarray
            2D array with data-type `numpy.float64`.
            Visible units.
        
        Returns
        -------
        tuple
            numpy.float64
                Scaled Kullback-Lieber divergence
                of the approximate posterior from
                the prior.
            numpy.float64
                Error between the visible units and
                their reconstruction.
        
        """
        eps = numpy.random.normal(loc=0., scale=1., size=(visible_units.shape[0], self.__nb_z))
        (_, z_mean, z_log_std_squared, _, _, reconstruction) = \
            self.forward_pass(visible_units, epsilon=eps)
        scaled_kld = self.__alpha*tls.kl_divergence(z_mean, z_log_std_squared)
        rec_error = tls.reconstruction_error(visible_units,
                                             reconstruction,
                                             self.__is_continuous)
        return (scaled_kld, rec_error)
    
    def weights_decay(self):
        """Computes the l2-norm weight decay.
        
        Returns
        -------
        numpy.float64
            L2-norm weight decay.
        
        """
        accumulation = 0.
        for key_1 in ['weights_recognition', 'weights_generation']:
            for key_2 in self.__parameters[key_1].keys():
                accumulation += numpy.sum(self.__parameters[key_1][key_2]**2)
        return 0.5*self.__weights_decay_p*accumulation
    
    def checking_activations(self, visible_units, title_hist_0, title_hist_1, title_hist_2,
                             path_hist_0, path_hist_1, path_hist_2, path_image):
        """Creates visualizations of the recognition network activations and saves them.
        
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
        (hidden_recognition, z_mean, z_log_std_squared) = self.recognition_network(visible_units)
        tls.histogram(hidden_recognition.flatten(),
                      title_hist_0,
                      path_hist_0)
        tls.histogram(z_mean.flatten(),
                      title_hist_1,
                      path_hist_1)
        tls.histogram(z_log_std_squared.flatten(),
                      title_hist_2,
                      path_hist_2)
        
        # `hidden_recognition_rescaled.dtype` is
        # equal to `numpy.float64`.
        hidden_recognition_rescaled = 255.*hidden_recognition/numpy.amax(hidden_recognition)
        
        # `hidden_recognition_uint8.dtype` is
        # equal to `numpy.uint8`.
        hidden_recognition_uint8 = numpy.round(hidden_recognition_rescaled).astype(numpy.uint8)
        tls.save_image(path_image,
                       hidden_recognition_uint8)
    
    def checking_p_1(self, key_type, key_layer):
        """Divides the parameter updates mean magnitude by the parameters mean magnitude.
        
        A single set of weights/biases is considered.
        To find this set in the dictionary that stores
        the parameters of the variational autoencoder,
        two keys are required.
        
        Parameters
        ----------
        key_type : str
            1st key. It is equal to either
            "weights_recognition" or "biases_recognition"
            if this set is in the recognition network. It
            is equal to either "weights_generation" or
            "biases_generation" if this set is in the
            generation network.
        key_layer : str
            2nd key. It is equal to either "l1",
            "mean" or "log_std_squared" if this set
            is in the recognition network. It is 
            equal to either "l1" or "mean" if this
            set is in the generation network.
        
        Returns
        -------
        numpy.float64
            Parameter updates mean magnitude divided by
            the parameters mean magnitude.
        
        """
        parameters = self.__parameters[key_type][key_layer].flatten()
        updates = self.__updates[key_type][key_layer].flatten()
        return numpy.mean(numpy.absolute(updates))/numpy.mean(numpy.absolute(parameters))
    
    def checking_p_2(self, key_type, key_layer, title, title_update, path, path_update):
        """Creates a histogram of parameters, a histogram of their update and saves the two histograms.
        
        A single set of weights/biases is considered.
        To find this set in the dictionary that stores
        the parameters of the variational autoencoder,
        two keys are required.
        
        Parameters
        ----------
        key_type : str
            1st key. It is equal to either
            "weights_recognition" or "biases_recognition"
            if this set is in the recognition network. It
            is equal to either "weights_generation" or
            "biases_generation" if this set is in the
            generation network.
        key_layer : str
            2nd key. It is equal to either "l1",
            "mean" or "log_std_squared" if this set
            is in the recognition network. It is 
            equal to either "l1" or "mean" if this
            set is in the generation network.
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
        tls.histogram(self.__parameters[key_type][key_layer].flatten(),
                      title,
                      path)
        tls.histogram(self.__updates[key_type][key_layer].flatten(),
                      title_update,
                      path_update)
    
    def checking_p_3(self, is_recognition, nb_display, height_image, width_image, nb_vertically, path):
        """Arranges the weight filters in a single RGB image and saves the single image.
        
        Parameters
        ----------
        is_recognition : bool
            If True, the weights in the 1st recognition
            layer are considered. Otherwise, the weights
            in the 2nd generation layer are considered.
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
        if is_recognition:
            weights = numpy.transpose(self.__parameters['weights_recognition']['l1'][:, 0:nb_display])
        else:
            weights = self.__parameters['weights_generation']['mean'][0:nb_display, :]
        tls.visualize_weights(weights,
                              height_image,
                              width_image,
                              nb_vertically,
                              path)


