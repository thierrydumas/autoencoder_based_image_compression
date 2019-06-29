"""A library that defines the class `IsolatedDecoder`."""

import tensorflow as tf

import eae.graph.components
import eae.graph.constants as csts
import tfutils.tfutils as tfuls


class IsolatedDecoder(object):
    """Isolated decoder class.
    
    The designation `isolated decoder` refers
    to the decoder of the entropy autoencoder alone.
    
    The attributes are the nodes we need
    to fetch while running the graph.
    
    """
    
    def __init__(self, batch_size, h_in, w_in, are_bin_widths_learned):
        """Builds the graph of the isolated decoder.
    
        Parameters
        ----------
        batch_size : int
            Size of the mini-batches.
        h_in : int
            Height of the images returned
            by the isolated decoder.
        w_in : int
            Width of the images returned
            by the isolated decoder.
        are_bin_widths_learned : bool
            Were the quantization bin widths learned
            during the autoencoder training?
        
        Raises
        ------
        ValueError
            If the height of the images returned by the isolated
            decoder is not divisible by the product of the three
            strides.
        ValueError
            If the width of the images returned by the isolated
            decoder is not divisible by the product of the three
            strides.
        
        """
        if h_in % csts.STRIDE_PROD != 0:
            raise ValueError('The height of the images returned by the isolated decoder is not divisible by the product of the three strides.')
        if w_in % csts.STRIDE_PROD != 0:
            raise ValueError('The width of the images returned by the isolated decoder is not divisible by the product of the three strides.')
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
        
        # Below is the reconstruction of the visible
        # units from the quantized latent variables.
        self.node_quantized_y = tf.placeholder(tf.float32,
                                               shape=(batch_size, h_in//csts.STRIDE_PROD, w_in//csts.STRIDE_PROD, csts.NB_MAPS_3))
        self.node_reconstruction = eae.graph.components.decoder(self.node_quantized_y,
                                                                are_bin_widths_learned)
        
        # Below is the backup of all variables.
        self.node_saver = tf.train.Saver()
    
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
            self.node_saver.restore(sess, path_to_restore)
        else:
            if tf.__version__.startswith('0'):
                tf.initialize_all_variables().run()
            else:
                tf.global_variables_initializer().run()


