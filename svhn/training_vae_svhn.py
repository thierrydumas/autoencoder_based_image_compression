"""A script to train a variational autoencoder.

The variational autoencoder is trained
on the SVHN training set.

"""

import argparse
import numpy
import os
import pickle
import time

import parsing.parsing
import svhn.svhn
import tools.tools as tls
from vae.VariationalAutoencoder import VariationalAutoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a variational autoencoder.')
    parser.add_argument('--alpha',
                        help='scaling coefficient',
                        type=parsing.parsing.float_strictly_positive,
                        default=1.,
                        metavar='')
    parser.add_argument('--nb_hidden',
                        help='number of recognition hidden units',
                        type=parsing.parsing.int_strictly_positive,
                        default=300,
                        metavar='')
    parser.add_argument('--nb_z',
                        help='number of latent variables',
                        type=parsing.parsing.int_strictly_positive,
                        default=25,
                        metavar='')
    parser.add_argument('--nb_epochs_training',
                        help='number of training epochs',
                        type=parsing.parsing.int_strictly_positive,
                        default=500,
                        metavar='')
    parser.add_argument('--batch_size',
                        help='size of the mini-batches',
                        type=parsing.parsing.int_strictly_positive,
                        default=100,
                        metavar='')
    args = parser.parse_args()
    path_to_training_data = 'svhn/results/training_data.npy'
    path_to_validation_data = 'svhn/results/validation_data.npy'
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    path_to_checking_a = 'vae/visualization/training/checking_activations/'
    path_to_checking_l = 'vae/visualization/training/checking_loss/'
    path_to_checking_p = 'vae/visualization/training/checking_parameters/'
    path_to_model = 'vae/results/vae_svhn.pkl'
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
    variational_ae = VariationalAutoencoder(nb_visible,
                                            args.nb_hidden,
                                            args.nb_z,
                                            True,
                                            args.alpha)
    scaled_kld = numpy.zeros((2, args.nb_epochs_training))
    rec_error = numpy.zeros((2, args.nb_epochs_training))
    w_decay = numpy.zeros((1, args.nb_epochs_training))
    mean_magnitude = numpy.zeros((5, args.nb_epochs_training))
    t_start = time.time()
    for i in range(args.nb_epochs_training):
        print('\nEpoch: {}'.format(i + 1))
        tuple_lr = variational_ae.learning_rate
        print('Learning rate (mean layer of the recognition network): {}'.format(tuple_lr[0]))
        print('Learning rate (other layers): {}'.format(tuple_lr[1]))
        (scaled_kld[0, i], rec_error[0, i]) = variational_ae.evaluation(training_portion_float64)
        (scaled_kld[1, i], rec_error[1, i]) = variational_ae.evaluation(validation_float64)
        w_decay[0, i] = variational_ae.weights_decay()
        print('Training scaled Kullback-Lieber divergence: {}'.format(scaled_kld[0, i]))
        print('Validation scaled Kullback-Lieber divergence: {}'.format(scaled_kld[1, i]))
        print('Training reconstruction error: {}'.format(rec_error[0, i]))
        print('Validation reconstruction error: {}'.format(rec_error[1, i]))
        print('L2-norm weight decay: {}'.format(w_decay[0, i]))
        
        # The training set is shuffled at the
        # beginning of each epoch.
        permutation = numpy.random.permutation(nb_training)
        for j in range(nb_batches):
            batch_uint8 = training_uint8[permutation[j*args.batch_size:(j + 1)*args.batch_size], :]
            batch_float64 = svhn.svhn.preprocess_svhn(batch_uint8,
                                                      mean_training,
                                                      std_training)
            variational_ae.training(batch_float64)
            if j == 4:
                mean_magnitude[0, i] = variational_ae.checking_p_1('weights_recognition', 'l1')
                mean_magnitude[1, i] = variational_ae.checking_p_1('weights_recognition', 'mean')
                mean_magnitude[2, i] = variational_ae.checking_p_1('weights_recognition', 'log_std_squared')
                mean_magnitude[3, i] = variational_ae.checking_p_1('weights_generation', 'l1')
                mean_magnitude[4, i] = variational_ae.checking_p_1('weights_generation', 'mean')
                print('Mean magnitude ratio in the recognition layer l1: {}'.format(mean_magnitude[0, i]))
                print('Mean magnitude ratio in the recognition layer mean: {}'.format(mean_magnitude[1, i]))
                print('Mean magnitude ratio in the recognition layer log std squared: {}'.format(mean_magnitude[2, i]))
                print('Mean magnitude ratio in the generation layer l1: {}'.format(mean_magnitude[3, i]))
                print('Mean magnitude ratio in the generation layer mean: {}'.format(mean_magnitude[4, i]))
        str_epoch = str(i + 1)
        if i == 4 or i == 49 or i == args.nb_epochs_training - 1:
            variational_ae.checking_p_2('weights_recognition',
                                        'l1',
                                        'Weights recognition l1 (epoch ' + str_epoch + ')',
                                        'Weights recognition l1 updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'wr_l1_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'wr_l1_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('weights_recognition',
                                        'mean',
                                        'Weights recognition mean (epoch ' + str_epoch + ')',
                                        'Weights recognition mean updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'wr_mean_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'wr_mean_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('weights_recognition',
                                        'log_std_squared',
                                        'Weights recognition log std squared (epoch ' + str_epoch + ')',
                                        'Weights recognition log std squared updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'wr_log_std_squared_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'wr_log_std_squared_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('weights_generation',
                                        'l1',
                                        'Weights generation l1 (epoch ' + str_epoch + ')',
                                        'Weights generation l1 updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'wg_l1_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'wg_l1_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('weights_generation',
                                        'mean',
                                        'Weights generation mean (epoch ' + str_epoch + ')',
                                        'Weights generation mean updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'wg_mean_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'wg_mean_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('biases_recognition',
                                        'l1',
                                        'Biases recognition l1 (epoch ' + str_epoch + ')',
                                        'Biases recognition l1 updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'br_l1_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'br_l1_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('biases_recognition',
                                        'mean',
                                        'Biases recognition mean (epoch ' + str_epoch + ')',
                                        'Biases recognition mean updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'br_mean_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'br_mean_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_2('biases_recognition',
                                        'log_std_squared',
                                        'Biases recognition log std squared (epoch ' + str_epoch + ')',
                                        'Biases recognition log std squared updates (epoch ' + str_epoch + ')',
                                        os.path.join(path_to_checking_p, 'br_log_std_squared_' + str_epoch + '.png'),
                                        os.path.join(path_to_checking_p, 'br_log_std_squared_updates_' + str_epoch + '.png'))
            variational_ae.checking_p_3(True,
                                        args.nb_hidden,
                                        32,
                                        32,
                                        10,
                                        os.path.join(path_to_checking_p, 'image_wr_l1_' + str_epoch + '.png'))
            variational_ae.checking_p_3(False,
                                        args.nb_hidden,
                                        32,
                                        32,
                                        10,
                                        os.path.join(path_to_checking_p, 'image_wg_mean_' + str_epoch + '.png'))
            variational_ae.checking_activations(validation_float64[0:args.batch_size, :],
                                                'Activations recognition l1 (epoch ' + str_epoch + ')',
                                                'Activations recognition mean (epoch ' + str_epoch + ')',
                                                'Activation recognition log std squared (epoch ' + str_epoch + ')',
                                                os.path.join(path_to_checking_a, 'ar_l1_' + str_epoch + '.png'),
                                                os.path.join(path_to_checking_a, 'ar_mean_' + str_epoch + '.png'),
                                                os.path.join(path_to_checking_a, 'ar_log_std_squared_' + str_epoch + '.png'),
                                                os.path.join(path_to_checking_a, 'image_ar_l1_' + str_epoch + '.png'))
    
    # The optional argument `dtype` in the
    # function `numpy.linspace` was introduced
    # in Numpy 1.9.0.
    x_values = numpy.linspace(1,
                              args.nb_epochs_training,
                              num=args.nb_epochs_training,
                              dtype=numpy.int32)
    tls.plot_graphs(x_values,
                    scaled_kld,
                    'epoch',
                    'scaled KL divergence of the approximate posterior from the prior',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of scaled KL divergence over epochs',
                    os.path.join(path_to_checking_l, 'scaled_kld.png'))
    tls.plot_graphs(x_values,
                    rec_error,
                    'epoch',
                    'reconstruction error',
                    ['training', 'validation'],
                    ['r', 'b'],
                    'Evolution of the reconstruction error over epochs',
                    os.path.join(path_to_checking_l, 'reconstruction_error.png'))
    tls.plot_graphs(x_values,
                    w_decay,
                    'epoch',
                    'weight decay',
                    ['l2-norm weight decay'],
                    ['b'],
                    'Evolution of the weight decay over epochs',
                    os.path.join(path_to_checking_l, 'weight_decay.png'))
    tls.plot_graphs(x_values,
                    mean_magnitude,
                    'epoch',
                    'mean magnitude ratio',
                    ['recognition l1', 'recognition mean', 'recognition log std squared', 'generation l1', 'generation mean'],
                    ['r', 'b', 'g', 'c', 'k'],
                    'Evolution of the mean magnitude ratio over epochs',
                    os.path.join(path_to_checking_p, 'mean_magnitude_ratio.png'))
    t_stop = time.time()
    nb_hours = int((t_stop - t_start)/3600)
    nb_minutes = int((t_stop - t_start)/60)
    print('\nTraining time: {0} hours and {1} minutes.'.format(nb_hours, nb_minutes - 60*nb_hours))
    
    # The protocol 2 provides backward compatibility
    # between Python 3 and Python 2.
    with open(path_to_model, 'wb') as file:
        pickle.dump(variational_ae, file, protocol=2)


