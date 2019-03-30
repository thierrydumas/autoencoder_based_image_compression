"""A script to train an entropy autoencoder

The entropy autoencoder is trained on
the ImageNet training set.

"""

import argparse
import numpy
import os
import tensorflow as tf
import time

import eae.batching
import eae.graph.constants as csts
import parsing.parsing
import tools.tools as tls
from eae.graph.EntropyAutoencoder import EntropyAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an entropy autoencoder.')
    parser.add_argument('bin_width_init',
                        help='value of the quantization bin widths at the beginning of the 1st training',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('gamma_scaling',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('idx_training',
                        help='training phase index',
                        type=parsing.parsing.int_positive)
    parser.add_argument('--learn_bin_widths',
                        help='if given, the quantization bin widths are learned',
                        action='store_true',
                        default=False)
    parser.add_argument('--nb_epochs_fitting',
                        help='number of fitting epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=1,
                        metavar='')
    parser.add_argument('--nb_epochs_training',
                        help='number of training epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=80,
                        metavar='')
    args = parser.parse_args()
    
    # The size of the mini-batches has to be equal
    # to the number of validation examples. This
    # restriction makes the online checks very fast.
    batch_size = 10
    nb_display = 5
    path_to_training_data = 'datasets/imagenet/results/training_data.npy'
    path_to_validation_data = 'datasets/imagenet/results/validation_data.npy'
    if args.learn_bin_widths:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init), tls.float_to_str(args.gamma_scaling))
    suffix_idx_training = '{0}/training_index_{1}'.format(suffix, args.idx_training + 1)
    path_to_checking_a = os.path.join('eae/visualization/training/checking_activations/',
                                      suffix_idx_training)
    if not os.path.isdir(path_to_checking_a):
        os.makedirs(path_to_checking_a)
    path_to_checking_l = os.path.join('eae/visualization/training/checking_loss/',
                                      suffix_idx_training)
    if not os.path.isdir(path_to_checking_l):
        os.makedirs(path_to_checking_l)
    path_to_checking_p = os.path.join('eae/visualization/training/checking_parameters/',
                                      suffix_idx_training)
    if not os.path.isdir(path_to_checking_p):
        os.makedirs(path_to_checking_p)
    path_to_directory_model = os.path.join('eae/results/',
                                           suffix)
    if not os.path.isdir(path_to_directory_model):
        os.makedirs(path_to_directory_model)
    if args.idx_training:
        path_to_nb_itvs_per_side_load = os.path.join(path_to_directory_model,
                                                     'nb_itvs_per_side_{}.pkl'.format(args.idx_training))
        path_to_restore = os.path.join(path_to_directory_model,
                                       'model_{}.ckpt'.format(args.idx_training))
    else:
        path_to_nb_itvs_per_side_load = ''
        path_to_restore = ''
    path_to_model = os.path.join(path_to_directory_model,
                                 'model_{}.ckpt'.format(args.idx_training + 1))
    path_to_meta = os.path.join(path_to_directory_model,
                                'model_{}.ckpt.meta'.format(args.idx_training + 1))
    path_to_nb_itvs_per_side_save = os.path.join(path_to_directory_model,
                                                 'nb_itvs_per_side_{}.pkl'.format(args.idx_training + 1))
    if os.path.isfile(path_to_model):
        print('"{}" already exists.'.format(path_to_model))
        print('Delete the model manually to retrain.')
        exit()
    elif os.path.isfile(path_to_meta):
        print('"{}" already exists.'.format(path_to_meta))
        print('Delete the metadata manually to retrain.')
        exit()
    
    # `training_uint8.dtype` is equal to `numpy.uint8`.
    training_uint8 = numpy.load(path_to_training_data)
    (nb_training, h_in, w_in, _) = training_uint8.shape
    nb_batches = tls.subdivide_set(nb_training, batch_size)
    print('Number of training examples: {}'.format(nb_training))
    print('Size of the mini-batches: {}'.format(batch_size))
    print('Number of mini-batches: {}'.format(nb_batches))
    
    # `validation_float32.dtype` and `training_portion_float32.dtype`
    # are equal to `numpy.float32`.
    validation_float32 = numpy.load(path_to_validation_data).astype(numpy.float32)
    if validation_float32.shape[0] != batch_size:
        raise ValueError('The number of validation examples is not equal to {}.'.format(batch_size))
    print('Number of validation examples: {}'.format(batch_size))
    training_portion_float32 = training_uint8[0:batch_size, :, :, :].astype(numpy.float32)
    entropy_ae = EntropyAutoencoder(batch_size,
                                    h_in,
                                    w_in,
                                    args.bin_width_init,
                                    args.gamma_scaling,
                                    path_to_nb_itvs_per_side_load,
                                    args.learn_bin_widths)
    
    # `difference_mean_entropy` stores the difference between
    # the mean entropy and the mean approximate entropy over
    # the training epochs.
    mean_approx_entropy = numpy.zeros((2, args.nb_epochs_training))
    mean_disc_entropy = numpy.zeros((2, args.nb_epochs_training))
    scaled_approx_entropy = numpy.zeros((2, args.nb_epochs_training))
    rec_error = numpy.zeros((2, args.nb_epochs_training))
    loss_density_approx = numpy.zeros((2, args.nb_epochs_training))
    difference_mean_entropy = numpy.zeros((2, args.nb_epochs_training))
    w_decay = numpy.zeros((1, args.nb_epochs_training))
    
    # The number of sampling points per unit
    # interval in the grid does not change
    # during the training whereas the number of
    # unit intervals in the right half of the
    # grid can increase during the training.
    nb_itvs_per_side = numpy.zeros((1, args.nb_epochs_training), dtype=numpy.int32)
    t_start = time.time()
    with tf.Session() as sess:
        entropy_ae.initialization(sess, path_to_restore)
        
        # The parameters of the piecewise linear functions
        # are only pre-trained at the beginning of the 1st
        # training.
        if not args.idx_training:
            
            # Note that, in Python 2.x, the loop
            # control variable of list comprehensions
            # leak into the surrounding scope. It is
            # no longer the case in Python 3.x.
            # The latent variables perturbed by uniform
            # noise are also called noisy latent variables.
            entropy_ae.checking_activations_1(sess,
                                              validation_float32,
                                              ['Noisy latent variables {} before the fitting'.format(index_map) for index_map in range(nb_display)],
                                              [os.path.join(path_to_checking_a, 'noisy_latent_{}_before_fitting.png'.format(index_map)) for index_map in range(nb_display)])
            print('\nThe preliminary fitting of the parameters of the piecewise linear functions starts.')
            eae.batching.preliminary_fitting(training_uint8,
                                             sess,
                                             entropy_ae,
                                             batch_size,
                                             args.nb_epochs_fitting)
            print('The preliminary fitting is completed.')
            entropy_ae.checking_activations_1(sess,
                                              validation_float32,
                                              ['Noisy latent variables {} after the fitting'.format(index_map) for index_map in range(nb_display)],
                                              [os.path.join(path_to_checking_a, 'noisy_latent_{}_after_fitting.png'.format(index_map)) for index_map in range(nb_display)])
        
        for i in range(args.nb_epochs_training):
            print('\nEpoch: {}'.format(i + 1))
            (mean_disc_entropy[0, i], scaled_approx_entropy[0, i], rec_error[0, i], loss_density_approx[0, i]) = \
                entropy_ae.evaluation(sess, training_portion_float32)
            mean_approx_entropy[0, i] = tls.convert_approx_entropy(scaled_approx_entropy[0, i],
                                                                   args.gamma_scaling,
                                                                   csts.NB_MAPS_3)
            difference_mean_entropy[0, i] = mean_disc_entropy[0, i] - mean_approx_entropy[0, i]
            (mean_disc_entropy[1, i], scaled_approx_entropy[1, i], rec_error[1, i], loss_density_approx[1, i]) = \
                entropy_ae.evaluation(sess, validation_float32)
            mean_approx_entropy[1, i] = tls.convert_approx_entropy(scaled_approx_entropy[1, i],
                                                                   args.gamma_scaling,
                                                                   csts.NB_MAPS_3)
            difference_mean_entropy[1, i] = mean_disc_entropy[1, i] - mean_approx_entropy[1, i]
            w_decay[0, i] = sess.run(entropy_ae.node_weight_decay)
            nb_itvs_per_side[0, i] = entropy_ae.get_nb_intervals_per_side()
            print('Training mean approximate entropy: {}'.format(mean_approx_entropy[0, i]))
            print('Validation mean approximate entropy: {}'.format(mean_approx_entropy[1, i]))
            print('Training mean entropy: {}'.format(mean_disc_entropy[0, i]))
            print('Validation mean entropy: {}'.format(mean_disc_entropy[1, i]))
            print('Training scaled cumulated approximate entropy: {}'.format(scaled_approx_entropy[0, i]))
            print('Validation scaled cumulated approximate entropy: {}'.format(scaled_approx_entropy[1, i]))
            print('Training reconstruction error: {}'.format(rec_error[0, i]))
            print('Validation reconstruction error: {}'.format(rec_error[1, i]))
            print('Training loss of density approximation: {}'.format(loss_density_approx[0, i]))
            print('Validation loss of density approximation: {}'.format(loss_density_approx[1, i]))
            print('Difference between the training mean entropy and the training mean approximate entropy: {}'.format(difference_mean_entropy[0, i]))
            print('Difference between the validation mean entropy and the validation mean approximate entropy: {}'.format(difference_mean_entropy[1, i]))
            print('L2-norm weight decay: {}'.format(w_decay[0, i]))
            print('Number of unit intervals in the right half of the grid: {}'.format(nb_itvs_per_side[0, i]))
            lr_eae = sess.run(entropy_ae.node_lr_eae)
            print('Learning rate for the parameters of the entropy autoencoder: {}'.format(round(lr_eae.item(), 6)))
            print('Number of updates of the parameters of the entropy autoencoder since the beginning of the 1st training: {}'.format(entropy_ae.get_global_step()))
            
            # The training set is shuffled at the
            # beginning of the function `run_epoch_training`.
            eae.batching.run_epoch_training(training_uint8,
                                            sess,
                                            entropy_ae,
                                            batch_size,
                                            nb_batches)
            str_epoch = str(i + 1)
            if i == 4 or i == 19 or i == args.nb_epochs_training - 1:
                entropy_ae.checking_activations_1(sess,
                                                  validation_float32,
                                                  ['Noisy latent variables {0} (epoch {1})'.format(index_map, str_epoch) for index_map in range(nb_display)],
                                                  [os.path.join(path_to_checking_a, 'noisy_latent_{0}_{1}.png'.format(index_map, str_epoch)) for index_map in range(nb_display)])
                entropy_ae.checking_activations_2(sess,
                                                  validation_float32,
                                                  [os.path.join(path_to_checking_a, 'image_latent_{0}_{1}.png'.format(index_map, str_epoch)) for index_map in range(2)])
                entropy_ae.checking_area_under_piecewise_linear_functions(sess,
                                                                          'Area under the piecewise linear functions (epoch ' + str_epoch + ')',
                                                                          os.path.join(path_to_checking_p, 'area_under_piecewise_linear_functions_' + str_epoch + '.png'))
                entropy_ae.checking_p_1('encoder',
                                        'weights_1',
                                        'Weights of the 1st convolutional layer (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'weights_1_' + str_epoch + '.png'))
                entropy_ae.checking_p_1('decoder',
                                        'weights_6',
                                        'Weights of the 3rd transpose convolutional layer (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'weights_6_' + str_epoch + '.png'))
                entropy_ae.checking_p_1('encoder',
                                        'gamma_1',
                                        'Weights of the 1st GDN (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'gamma_1_' + str_epoch + '.png'))
                entropy_ae.checking_p_1('encoder',
                                        'beta_1',
                                        'Additive coefficients of the 1st GDN (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'beta_1_' + str_epoch + '.png'))
                if args.learn_bin_widths:
                    entropy_ae.checking_p_1('piecewise_linear_function',
                                            'bin_widths',
                                            'Quantization bin widths (epoch ' + str_epoch + ')',
                                            os.path.join(path_to_checking_p, 'bin_widths_' + str_epoch + '.png'))
                entropy_ae.checking_p_2(True,
                                        8,
                                        os.path.join(path_to_checking_p, 'image_weights_1_' + str_epoch + '.png'))
                entropy_ae.checking_p_2(False,
                                        8,
                                        os.path.join(path_to_checking_p, 'image_weights_6_' + str_epoch + '.png'))
                entropy_ae.checking_p_3('encoder',
                                        1,
                                        os.path.join(path_to_checking_p, 'image_gamma_1_' + str_epoch + '.png'))
            entropy_ae.save(sess,
                            path_to_model,
                            path_to_nb_itvs_per_side_save)
    
    # The optional argument `dtype` in
    # the function `numpy.linspace` was
    # introduced in Numpy 1.9.0.
    x_values = numpy.linspace(1,
                              args.nb_epochs_training,
                              num=args.nb_epochs_training,
                              dtype=numpy.int32)
    tls.plot_graphs(x_values,
                    mean_approx_entropy,
                    'epoch',
                    'mean approximate entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the mean approximate entropy over epochs',
                    os.path.join(path_to_checking_l, 'mean_approximate_entropy.png'))
    tls.plot_graphs(x_values,
                    mean_disc_entropy,
                    'epoch',
                    'mean entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the mean entropy over epochs',
                    os.path.join(path_to_checking_l, 'mean_entropy.png'))
    tls.plot_graphs(x_values,
                    scaled_approx_entropy,
                    'epoch',
                    'scaled cumulated approximate entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the scaled cumulated approximate entropy over epochs',
                    os.path.join(path_to_checking_l, 'scaled_approximate_entropy.png'))
    tls.plot_graphs(x_values,
                    rec_error,
                    'epoch',
                    'reconstruction error',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the reconstruction error over epochs',
                    os.path.join(path_to_checking_l, 'reconstruction_error.png'))
    tls.plot_graphs(x_values,
                    loss_density_approx,
                    'epoch',
                    'loss of density approximation',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the loss of density approximation over epochs',
                    os.path.join(path_to_checking_l, 'loss_density_approximation.png'))
    tls.plot_graphs(x_values,
                    difference_mean_entropy,
                    'epoch',
                    'difference in mean entropy',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the difference in mean entropy over epochs',
                    os.path.join(path_to_checking_l, 'difference_mean_entropy.png'))
    tls.plot_graphs(x_values,
                    w_decay,
                    'epoch',
                    'weight decay',
                    ['l2-norm weight decay'],
                    ['b'],
                    'Evolution of the weight decay over epochs',
                    os.path.join(path_to_checking_l, 'weight_decay.png'))
    tls.plot_graphs(x_values,
                    nb_itvs_per_side,
                    'epoch',
                    'number of unit intervals per side',
                    ['grid symmetrical about 0'],
                    ['b'],
                    'Evolution of the number of unit intervals per side over epochs',
                    os.path.join(path_to_checking_p, 'nb_intervals_per_side.png'))
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nTraining time: {0} hours and {1} minutes.'.format(nb_hours, nb_minutes - 60*nb_hours))


