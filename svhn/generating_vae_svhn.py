"""A script to generate from a trained variational autoencoder."""

import argparse
import numpy
import os
import pickle

import parsing.parsing
import tools.tools as tls

def generating_vae_svhn(z_reference, mean_training, std_training, variational_ae, nb_interpolations, path):
    """Interpolates between the reference points in the latent space and generates from the interpolation.
    
    Parameters
    ----------
    z_reference : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Reference points in the latent space.
        `reference_z[i, :]` contains the ith
        reference point.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each pixel
        over all training images.
    variational_ae : VariationalAutoencoder
        Trained variational autoencoder.
    nb_interpolations : int
        Number of interpolations between two
        reference points in the latent space.
    path : str
        Path to the saved image of the generation.
        The path must end with ".png".
    
    Raises
    ------
    ValueError
        If `z_reference.shape[1]` is not equal
        to `variational_ae.nb_z`.
    
    """
    (nb_reference_points, width) = z_reference.shape
    if width != variational_ae.nb_z:
        raise ValueError('`z_reference.shape[1]` is not equal to `variational_ae.nb_z`.')
    
    # Between two reference points in the latent
    # space, `nb_interpolations` points are
    # calculated by interpolation.
    nb_points = nb_reference_points*nb_interpolations
    z = numpy.zeros((nb_points, variational_ae.nb_z))
    for i in range(variational_ae.nb_z):
        for j in range(nb_reference_points):
            z[j*nb_interpolations:(j + 1)*nb_interpolations, i] = \
                numpy.linspace(z_reference[j, i],
                               z_reference[(j + 1) % nb_reference_points, i],
                               num=nb_interpolations)
    generation_float64 = variational_ae.generation_network(z)[1]
    gen_rescaled_float64 = generation_float64*std_training + \
        numpy.tile(mean_training, (nb_points, 1))
    generation_uint8 = tls.cast_float_to_uint8(gen_rescaled_float64)
    tls.visualize_rows(generation_uint8,
                       32,
                       32,
                       nb_reference_points,
                       path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates from a trained variational autoencoder.')
    parser.add_argument('--nb_reference_points',
                        help='number of reference points in the latent space',
                        type=parsing.parsing.int_strictly_positive,
                        default=8,
                        metavar='')
    parser.add_argument('--nb_interpolations',
                        help='number of interpolations between two reference points',
                        type=parsing.parsing.int_strictly_positive,
                        default=10,
                        metavar='')
    parser.add_argument('--ball_radius',
                        help='radius of the ball containing the reference points',
                        type=parsing.parsing.float_strictly_positive,
                        default=3.,
                        metavar='')
    args = parser.parse_args()
    
    path_to_mean_training = 'svhn/results/mean_training.npy'
    path_to_std_training = 'svhn/results/std_training.npy'
    path_to_checking_g = 'vae/visualization/test/checking_generating/'
    path_to_model = 'vae/results/vae_svhn.pkl'
    with open(path_to_model, 'rb') as file:
        variational_ae = pickle.load(file)
    
    # `mean_training.dtype` and `std_training.dtype`
    # are equal to `numpy.float64`.
    mean_training = numpy.load(path_to_mean_training)
    std_training = numpy.load(path_to_std_training)
    z_reference = numpy.random.uniform(low=-args.ball_radius,
                                       high=args.ball_radius,
                                       size=(args.nb_reference_points, variational_ae.nb_z))
    generating_vae_svhn(z_reference,
                        mean_training,
                        std_training,
                        variational_ae,
                        args.nb_interpolations,
                        os.path.join(path_to_checking_g, 'generation.png'))


