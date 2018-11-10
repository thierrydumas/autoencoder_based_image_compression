"""A script to train an entropy autoencoder.

The entropy autoencoder is trained on
the SVHN training set.

"""

import argparse
import numpy
import os
import pickle
import time

import eae.eae_utils as eaeuls
import parsing.parsing
import svhn.svhn
import tools.tools as tls
from eae.EntropyAutoencoder import EntropyAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains an entropy autoencoder.')
    parser.add_argument('bin_width_init',
                        help='value of the quantization bin width at the beginning of the training',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('gamma',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive)
    parser.add_argument('--learn_bin_width',
                        help='if given, the quantization bin width is learned',
                        action='store_true',
                        default=False)
    parser.add_argument('--nb_hidden',
                        help='number of encoder hidden units',
                        type=parsing.parsing.int_strictly_positive,
                        default=300,
                        metavar='')
    parser.add_argument('--nb_y',
                        help='number of latent variables',
                        type=parsing.parsing.int_strictly_positive,
                        default=200,
                        metavar='')
    parser.add_argument('--nb_epochs_fitting',
                        help='number of fitting epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=1,
                        metavar='')
    parser.add_argument('--nb_epochs_training',
                        help='number of training epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=800,
                        metavar='')
    parser.add_argument('--batch_size',
                        help='size of the mini-batches',
                        type=parsing.parsing.int_strictly_positive,
                        default=250,
                        metavar='')
    args = parser.parse_args()
    path_to_training_data = 'svhn/results/training_data.npy'
    path_to_validation_data = 'svhn/results/validation_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    if args.learn_bin_width:
        suffix = 'learning_bw_{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                              tls.float_to_str(args.gamma))
    else:
        suffix = '{0}_{1}'.format(tls.float_to_str(args.bin_width_init),
                                  tls.float_to_str(args.gamma))
    path_to_checking_a = 'eae/visualization/training/checking_activations/' + suffix + '/'
    path_to_checking_l = 'eae/visualization/training/checking_loss/' + suffix + '/'
    path_to_checking_p = 'eae/visualization/training/checking_parameters/' + suffix + '/'
    path_to_model = 'eae/results/eae_svhn_' + suffix + '.pkl'
    if os.path.isfile(path_to_model):
        print('"{}" already exists.'.format(path_to_model))
        print('Delete it manually to retrain.')
        exit()
    
    # `training_uint8.dtype` is equal to `numpy.uint8`.
    training_uint8 = numpy.load(path_to_training_data)
    (nb_training, nb_visible) = training_uint8.shape
    nb_batches = tls.subdivide_set(nb_training, args.batch_size)
    print('Number of training examples: {}'.format(nb_training))
    print('Size of the mini-batches: {}'.format(args.batch_size))
    print('Number of mini-batches: {}'.format(nb_batches))
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    
    # `validation_float64.dtype` and `training_portion_float64.dtype`
    # are equal to `numpy.float64`.
    validation_float64 = svhn.svhn.preprocess_svhn(numpy.load(path_to_validation_data),
                                                   mean_training,
                                                   std_training)
    nb_validation = validation_float64.shape[0]
    assert nb_validation >= args.batch_size, \
        'The number of validation examples is not larger than {}.'.format(args.batch_size)
    print('Number of validation examples: {}'.format(nb_validation))
    training_portion_float64 = svhn.svhn.preprocess_svhn(training_uint8[0:nb_validation, :],
                                                         mean_training,
                                                         std_training)
    entropy_ae = EntropyAutoencoder(nb_visible,
                                    args.nb_hidden,
                                    args.nb_y,
                                    args.bin_width_init,
                                    args.gamma,
                                    args.learn_bin_width)
    
    # The latent variables perturbed by uniform
    # noise are also called noisy latent variables.
    entropy_ae.checking_activations(validation_float64[0:args.batch_size, :],
                                    'Activations encoder l1 before the fitting',
                                    'Activations encoder latent before the fitting',
                                    'Activations encoder noisy latent before the fitting',
                                    path_to_checking_a + 'ae_l1_before_fitting.png',
                                    path_to_checking_a + 'ae_latent_before_fitting.png',
                                    path_to_checking_a + 'ae_noisy_latent_before_fitting.png',
                                    path_to_checking_a + 'image_dead_zone_before_fitting.png')
    print('\nThe preliminary fitting of the parameters of the piecewise linear function starts.')
    eaeuls.preliminary_fitting(training_uint8,
                               mean_training,
                               std_training,
                               entropy_ae,
                               args.batch_size,
                               args.nb_epochs_fitting)
    print('The preliminary fitting is completed.')
    entropy_ae.checking_activations(validation_float64[0:args.batch_size, :],
                                    'Activations encoder l1 after the fitting',
                                    'Activations encoder latent after the fitting',
                                    'Activations encoder noisy latent after the fitting',
                                    path_to_checking_a + 'ae_l1_after_fitting.png',
                                    path_to_checking_a + 'ae_latent_after_fitting.png',
                                    path_to_checking_a + 'ae_noisy_latent_after_fitting.png',
                                    path_to_checking_a + 'image_dead_zone_after_fitting.png')
    
    approx_entropy = numpy.zeros((2, args.nb_epochs_training))
    disc_entropy = numpy.zeros((2, args.nb_epochs_training))
    scaled_approx_entropy = numpy.zeros((2, args.nb_epochs_training))
    rec_error = numpy.zeros((2, args.nb_epochs_training))
    loss_density_approx = numpy.zeros((2, args.nb_epochs_training))
    nb_dead = numpy.zeros((2, args.nb_epochs_training), dtype=numpy.int32)
    difference_entropy = numpy.zeros((2, args.nb_epochs_training))
    w_decay = numpy.zeros((1, args.nb_epochs_training))
    
    # The number of sampling points per unit interval
    # in the grid does not change during the training
    # whereas the number of unit intervals in the right
    # half of the grid can increase during the training.
    nb_itvs_per_side = numpy.zeros((1, args.nb_epochs_training), dtype=numpy.int32)
    memory_bin_width = numpy.zeros((1, args.nb_epochs_training))
    mean_magnitude = numpy.zeros((4, args.nb_epochs_training))
    t_start = time.time()
    for i in range(args.nb_epochs_training):
        print('\nEpoch: {}'.format(i + 1))
        (approx_entropy[0, i], disc_entropy[0, i], scaled_approx_entropy[0, i],
            rec_error[0, i], loss_density_approx[0, i], nb_dead[0, i]) = \
            entropy_ae.evaluation(training_portion_float64)
        difference_entropy[0, i] = disc_entropy[0, i] - approx_entropy[0, i]
        (approx_entropy[1, i], disc_entropy[1, i], scaled_approx_entropy[1, i],
            rec_error[1, i], loss_density_approx[1, i], nb_dead[1, i]) = \
            entropy_ae.evaluation(validation_float64)
        difference_entropy[1, i] = disc_entropy[1, i] - approx_entropy[1, i]
        w_decay[0, i] = entropy_ae.weights_decay()
        nb_itvs_per_side[0, i] = entropy_ae.nb_intervals_per_side
        memory_bin_width[0, i] = entropy_ae.bin_width
        print('Training approximate entropy: {}'.format(approx_entropy[0, i]))
        print('Validation approximate entropy: {}'.format(approx_entropy[1, i]))
        print('Training entropy: {}'.format(disc_entropy[0, i]))
        print('Validation entropy: {}'.format(disc_entropy[1, i]))
        print('Training scaled approximate entropy: {}'.format(scaled_approx_entropy[0, i]))
        print('Validation scaled approximate entropy: {}'.format(scaled_approx_entropy[1, i]))
        print('Training reconstruction error: {}'.format(rec_error[0, i]))
        print('Validation reconstruction error: {}'.format(rec_error[1, i]))
        print('Training loss of density approximation: {}'.format(loss_density_approx[0, i]))
        print('Validation loss of density approximation: {}'.format(loss_density_approx[1, i]))
        print('Training number of dead quantized latent variables: {}'.format(nb_dead[0, i]))
        print('Validation number of dead quantized latent variables: {}'.format(nb_dead[1, i]))
        print('Difference between the training entropy and the training approximate entropy: {}'.format(difference_entropy[0, i]))
        print('Difference between the validation entropy and the validation approximate entropy: {}'.format(difference_entropy[1, i]))
        print('L2-norm weight decay: {}'.format(w_decay[0, i]))
        print('Area under the piecewise linear function: {}'.format(entropy_ae.area_under_piecewise_linear_function()))
        print('Number of unit intervals in the right half of the grid: {}'.format(nb_itvs_per_side[0, i]))
        print('Quantization bin width: {}'.format(memory_bin_width[0, i]))
        
        # The training set is shuffled at the
        # beginning of each epoch.
        permutation = numpy.random.permutation(nb_training)
        for j in range(nb_batches):
            batch_uint8 = training_uint8[permutation[j*args.batch_size:(j + 1)*args.batch_size], :]
            batch_float64 = svhn.svhn.preprocess_svhn(batch_uint8,
                                                      mean_training,
                                                      std_training)
            
            # The parameters of the piecewise linear function
            # and the parameters of the entropy autoencoder
            # are optimized alternatively.
            # For a given training batch, the method
            # `training_fct` of class `EntropyAutoencoder`
            # is called before the method `training_eae_bw`
            # as the first mentioned method verifies whether
            # the condition of expansion is met.
            entropy_ae.training_fct(batch_float64)
            entropy_ae.training_eae_bw(batch_float64)
            if j == 4:
                mean_magnitude[0, i] = entropy_ae.checking_p_1('weights_encoder', 'l1')
                mean_magnitude[1, i] = entropy_ae.checking_p_1('weights_encoder', 'latent')
                mean_magnitude[2, i] = entropy_ae.checking_p_1('weights_decoder', 'l1')
                mean_magnitude[3, i] = entropy_ae.checking_p_1('weights_decoder', 'mean')
                print('Mean magnitude ratio in the encoder layer l1: {}'.format(mean_magnitude[0, i]))
                print('Mean magnitude ratio in the encoder layer latent: {}'.format(mean_magnitude[1, i]))
                print('Mean magnitude ratio in the decoder layer l1: {}'.format(mean_magnitude[2, i]))
                print('Mean magnitude ratio in the decoder layer mean: {}'.format(mean_magnitude[3, i]))
        str_epoch = str(i + 1)
        if i == 4 or i == 49 or i == args.nb_epochs_training - 1:
            entropy_ae.checking_p_2('weights_encoder',
                                    'l1',
                                    'Weights encoder l1 (epoch ' + str_epoch + ')',
                                    'Weights encoder l1 updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'we_l1_' + str_epoch + '.png',
                                    path_to_checking_p + 'we_l1_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_2('weights_encoder',
                                    'latent',
                                    'Weights encoder latent (epoch ' + str_epoch + ')',
                                    'Weights encoder latent updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'we_latent_' + str_epoch + '.png',
                                    path_to_checking_p + 'we_latent_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_2('weights_decoder',
                                    'l1',
                                    'Weights decoder l1 (epoch ' + str_epoch + ')',
                                    'Weights decoder l1 updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'wd_l1_' + str_epoch + '.png',
                                    path_to_checking_p + 'wd_l1_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_2('weights_decoder',
                                    'mean',
                                    'Weights decoder mean (epoch ' + str_epoch + ')',
                                    'Weights decoder mean updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'wd_mean_' + str_epoch + '.png',
                                    path_to_checking_p + 'wd_mean_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_2('biases_encoder',
                                    'l1',
                                    'Biases encoder l1 (epoch ' + str_epoch + ')',
                                    'Biases encoder l1 updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'be_l1_' + str_epoch + '.png',
                                    path_to_checking_p + 'be_l1_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_2('biases_encoder',
                                    'latent',
                                    'Biases encoder latent (epoch ' + str_epoch + ')',
                                    'Biases encoder latent updates (epoch ' + str_epoch + ')',
                                    path_to_checking_p + 'be_latent_' + str_epoch + '.png',
                                    path_to_checking_p + 'be_latent_updates_' + str_epoch + '.png')
            entropy_ae.checking_p_3(True,
                                    args.nb_hidden,
                                    32,
                                    32,
                                    10,
                                    path_to_checking_p + 'image_we_l1_' + str_epoch + '.png')
            entropy_ae.checking_p_3(False,
                                    args.nb_hidden,
                                    32,
                                    32,
                                    10,
                                    path_to_checking_p + 'image_wd_mean_' + str_epoch + '.png')
            entropy_ae.checking_activations(validation_float64[0:args.batch_size, :],
                                            'Activations encoder l1 (epoch ' + str_epoch + ')',
                                            'Activations encoder latent (epoch ' + str_epoch + ')',
                                            'Activations encoder noisy latent (epoch ' + str_epoch + ')',
                                            path_to_checking_a + 'ae_l1_' + str_epoch + '.png',
                                            path_to_checking_a + 'ae_latent_' + str_epoch + '.png',
                                            path_to_checking_a + 'ae_noisy_latent_' + str_epoch + '.png',
                                            path_to_checking_a + 'image_dead_zone_' + str_epoch + '.png')
    
    # The optional argument `dtype` in the
    # function `numpy.linspace` was introduced
    # in Numpy 1.9.0.
    x_values = numpy.linspace(1,
                              args.nb_epochs_training,
                              num=args.nb_epochs_training,
                              dtype=numpy.int32)
    tls.plot_graphs(x_values,
                    approx_entropy,
                    'epoch',
                    'approximate entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the approximate entropy over epochs',
                    path_to_checking_l + 'approximate_entropy.png')
    tls.plot_graphs(x_values,
                    disc_entropy,
                    'epoch',
                    'entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the entropy over epochs',
                    path_to_checking_l + 'entropy.png')
    tls.plot_graphs(x_values,
                    scaled_approx_entropy,
                    'epoch',
                    'scaled approximate entropy of the quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the scaled approximate entropy over epochs',
                    path_to_checking_l + 'scaled_approximate_entropy.png')
    tls.plot_graphs(x_values,
                    rec_error,
                    'epoch',
                    'reconstruction error',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the reconstruction error over epochs',
                    path_to_checking_l + 'reconstruction_error.png')
    tls.plot_graphs(x_values,
                    loss_density_approx,
                    'epoch',
                    'loss of density approximation',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the loss of density approximation over epochs',
                    path_to_checking_l + 'loss_density_approximation.png')
    tls.plot_graphs(x_values,
                    difference_entropy,
                    'epoch',
                    'difference in entropy',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the difference in entropy over epochs',
                    path_to_checking_l + 'difference_entropy.png')
    tls.plot_graphs(x_values,
                    w_decay,
                    'epoch',
                    'weight decay',
                    ['L2-norm weight decay'],
                    ['b'],
                    'Evolution of the weight decay over epochs',
                    path_to_checking_l + 'weight_decay.png')
    tls.plot_graphs(x_values,
                    nb_dead,
                    'epoch',
                    'number of dead quantized latent variables',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the number of dead quantized latent variables over epochs',
                    path_to_checking_a + 'nb_dead.png')
    tls.plot_graphs(x_values,
                    nb_itvs_per_side,
                    'epoch',
                    'number of unit intervals per side',
                    ['grid symmetrical about 0'],
                    ['b'],
                    'Evolution of the number of unit intervals per side over epochs',
                    path_to_checking_p + 'nb_intervals_per_side.png')
    tls.plot_graphs(x_values,
                    mean_magnitude,
                    'epoch',
                    'mean magnitude ratio',
                    ['encoder l1', 'encoder latent', 'decoder l1', 'decoder mean'],
                    ['r', 'b', 'g', 'c'],
                    'Evolution of the mean magnitude ratio over epochs',
                    path_to_checking_p + 'mean_magnitude_ratio.png')
    if args.learn_bin_width:
        tls.plot_graphs(x_values,
                        memory_bin_width,
                        'epoch',
                        'quantization bin width',
                        ['learned bin width'],
                        ['b'],
                        'Evolution of the quantization bin width over epochs',
                        path_to_checking_p + 'bin_width.png')
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nTraining time: {0} hours and {1} minutes.'.format(nb_hours, nb_minutes - 60*nb_hours))
    
    # The protocol 2 provides backward compatibility
    # between Python 3 and Python 2.
    with open(path_to_model, 'wb') as file:
        pickle.dump(entropy_ae, file, protocol=2)


