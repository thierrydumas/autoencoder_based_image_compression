"""A library that contains components used both in the class `EntropyAutoencoder` and in the class `IsolatedDecoder`."""

import tensorflow as tf

import eae.graph.constants as csts
import tf_utils.tf_utils as tfuls

# The functions are sorted in
# alphabetic order.

def decoder(y_tilde, are_bin_widths_learned):
    """Computes a reconstruction of the visible units from the noisy latent variables.
    
    Parameters
    ----------
    y_tilde : Tensor
        4D tensor with data-type `tf.float32`.
        Latent variables perturbed by uniform noise.
    are_bin_widths_learned : bool
        Are the quantization bin widths learned?
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Reconstruction of the visible units.
    
    """
    with tf.variable_scope('decoder', reuse=True):
        if not are_bin_widths_learned:
            gamma_4 = tf.get_variable('gamma_4', dtype=tf.float32)
            beta_4 = tf.get_variable('beta_4', dtype=tf.float32)
        weights_4 = tf.get_variable('weights_4', dtype=tf.float32)
        biases_4 = tf.get_variable('biases_4', dtype=tf.float32)
        gamma_5 = tf.get_variable('gamma_5', dtype=tf.float32)
        beta_5 = tf.get_variable('beta_5', dtype=tf.float32)
        weights_5 = tf.get_variable('weights_5', dtype=tf.float32)
        biases_5 = tf.get_variable('biases_5', dtype=tf.float32)
        gamma_6 = tf.get_variable('gamma_6', dtype=tf.float32)
        beta_6 = tf.get_variable('beta_6', dtype=tf.float32)
        weights_6 = tf.get_variable('weights_6', dtype=tf.float32)
    
    # The shape of `y_tilde` does not change
    # while running the graph. Therefore, the
    # static shape of `y_tilde` is used.
    shape_y_tilde = y_tilde.get_shape().as_list()
    batch_size = shape_y_tilde[0]
    height_y_tilde = shape_y_tilde[1]
    width_y_tilde = shape_y_tilde[2]
    
    # If the condition is True, `input_transpose_conv_1`
    # is a reference to `y_tilde`.
    if are_bin_widths_learned:
        input_transpose_conv_1 = y_tilde
    else:
        input_transpose_conv_1 = tfuls.inverse_gdn(y_tilde,
                                                   gamma_4,
                                                   beta_4)
    
    # The shape of `weights_4` does not change
    # while running the graph. Therefore, the
    # static shape of `weights_4` is used.
    transpose_conv_1 = tf.nn.conv2d_transpose(input_transpose_conv_1,
                                              weights_4,
                                              [batch_size, csts.STRIDE_3*height_y_tilde, csts.STRIDE_3*width_y_tilde, weights_4.get_shape().as_list()[2]],
                                              strides=[1, csts.STRIDE_3, csts.STRIDE_3, 1],
                                              padding='SAME')
    igdn_2 = tfuls.inverse_gdn(tf.nn.bias_add(transpose_conv_1, biases_4),
                               gamma_5,
                               beta_5)
    transpose_conv_2 = tf.nn.conv2d_transpose(igdn_2,
                                              weights_5,
                                              [batch_size, csts.STRIDE_3*csts.STRIDE_2*height_y_tilde, csts.STRIDE_3*csts.STRIDE_2*width_y_tilde, weights_5.get_shape().as_list()[2]],
                                              strides=[1, csts.STRIDE_2, csts.STRIDE_2, 1],
                                              padding='SAME')
    igdn_3 = tfuls.inverse_gdn(tf.nn.bias_add(transpose_conv_2, biases_5),
                               gamma_6,
                               beta_6)
    transpose_conv_3 = tf.nn.conv2d_transpose(igdn_3,
                                              weights_6,
                                              [batch_size, csts.STRIDE_PROD*height_y_tilde, csts.STRIDE_PROD*width_y_tilde, weights_6.get_shape().as_list()[2]],
                                              strides=[1, csts.STRIDE_1, csts.STRIDE_1, 1],
                                              padding='SAME')
    return transpose_conv_3

def encoder(visible_units, are_bin_widths_learned):
    """Computes the latent variables from the visible units.
    
    Parameters
    ----------
    visible_units : Tensor
        4D tensor with data-type `tf.float32`.
        Visible units.
    are_bin_widths_learned : bool
        Are the quantization bin widths learned?
    
    Returns
    -------
    Tensor
        4D tensor with data-type `tf.float32`.
        Latent variables.
    
    """
    with tf.variable_scope('encoder', reuse=True):
        weights_1 = tf.get_variable('weights_1', dtype=tf.float32)
        biases_1 = tf.get_variable('biases_1', dtype=tf.float32)
        gamma_1 = tf.get_variable('gamma_1', dtype=tf.float32)
        beta_1 = tf.get_variable('beta_1', dtype=tf.float32)
        weights_2 = tf.get_variable('weights_2', dtype=tf.float32)
        biases_2 = tf.get_variable('biases_2', dtype=tf.float32)
        gamma_2 = tf.get_variable('gamma_2', dtype=tf.float32)
        beta_2 = tf.get_variable('beta_2', dtype=tf.float32)
        weights_3 = tf.get_variable('weights_3', dtype=tf.float32)
        biases_3 = tf.get_variable('biases_3', dtype=tf.float32)
        if not are_bin_widths_learned:
            gamma_3 = tf.get_variable('gamma_3', dtype=tf.float32)
            beta_3 = tf.get_variable('beta_3', dtype=tf.float32)
    
    conv_1 = tf.nn.conv2d(visible_units,
                          weights_1,
                          strides=[1, csts.STRIDE_1, csts.STRIDE_1, 1],
                          padding='SAME')
    gdn_1 = tfuls.gdn(tf.nn.bias_add(conv_1, biases_1),
                      gamma_1,
                      beta_1)
    conv_2 = tf.nn.conv2d(gdn_1,
                          weights_2,
                          strides=[1, csts.STRIDE_2, csts.STRIDE_2, 1],
                          padding='SAME')
    gdn_2 = tfuls.gdn(tf.nn.bias_add(conv_2, biases_2),
                      gamma_2,
                      beta_2)
    conv_3 = tf.nn.conv2d(gdn_2,
                          weights_3,
                          strides=[1, csts.STRIDE_3, csts.STRIDE_3, 1],
                          padding='SAME')
    if are_bin_widths_learned:
        return tf.nn.bias_add(conv_3, biases_3)
    else:
        return tfuls.gdn(tf.nn.bias_add(conv_3, biases_3),
                         gamma_3,
                         beta_3)

def weight_l2_norm():
    """Computes the cumulated weight l2-norm.
    
    Returns
    -------
    Tensor
        0D tensor with data-type `tf.float32`.
        Cumulated weight l2-norm.
    
    """
    with tf.variable_scope('encoder', reuse=True):
        weights_1 = tf.get_variable('weights_1', dtype=tf.float32)
        weights_2 = tf.get_variable('weights_2', dtype=tf.float32)
        weights_3 = tf.get_variable('weights_3', dtype=tf.float32)
    with tf.variable_scope('decoder', reuse=True):
        weights_4 = tf.get_variable('weights_4', dtype=tf.float32)
        weights_5 = tf.get_variable('weights_5', dtype=tf.float32)
        weights_6 = tf.get_variable('weights_6', dtype=tf.float32)
    return tf.nn.l2_loss(weights_1) \
           + tf.nn.l2_loss(weights_2) \
           + tf.nn.l2_loss(weights_3) \
           + tf.nn.l2_loss(weights_4) \
           + tf.nn.l2_loss(weights_5) \
           + tf.nn.l2_loss(weights_6)


