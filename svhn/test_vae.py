"""A script to test two methods of class `VariationalAutoencoder`."""

import argparse
import numpy

from vae.VariationalAutoencoder import VariationalAutoencoder


class TesterVariationalAutoencoder(object):
    """Class for testing two methods of class `VariationalAutoencoder`."""
    
    def test_backpropagation(self):
        """Tests the method `backpropagation`.
        
        The method `backpropagation` computes the gradients
        of the variational autoencoder parameters. It is
        tested via gradient checking, see
        <http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization>.
        
        Six histograms are saved in the folder
        "vae/pseudo_visualization/backpropagation/gaussian/"
        and six others are saved in the folder
        "vae/pseudo_visualization/backpropagation/bernoulli/".
        The test is successful if, in each
        histogram, the absolute values are
        smaller than 1.e-9.
        
        """
        batch_size = 10
        nb_visible = 15
        nb_hidden = 10
        nb_z = 2
        
        # 1st case: each visible unit is modeled
        # as a continuous random variable with
        # Gaussian probability density function.
        visible_units = numpy.random.normal(loc=0.,
                                            scale=1.,
                                            size=(batch_size, nb_visible))
        variational_ae_0 = VariationalAutoencoder(nb_visible,
                                                  nb_hidden,
                                                  nb_z,
                                                  True,
                                                  0.3)
        variational_ae_0.backpropagation(visible_units,
                                         is_checking=True,
                                         path_to_checking_g='vae/pseudo_visualization/backpropagation/gaussian/')
        
        # 2nd case: each visible unit is modeled
        # as the probability of activation of a
        # Bernoulli random variable.
        visible_units = numpy.random.uniform(low=0.,
                                             high=1.,
                                             size=(batch_size, nb_visible))
        variational_ae_1 = VariationalAutoencoder(nb_visible,
                                                  nb_hidden,
                                                  nb_z,
                                                  False,
                                                  0.3)
        variational_ae_1.backpropagation(visible_units,
                                         is_checking=True,
                                         path_to_checking_g='vae/pseudo_visualization/backpropagation/bernoulli/')
    
    def test_training(self):
        """Tests the method `training`.
        
        The test is successful if the reconstruction
        error after the training is much smaller than
        the reconstruction error before the training.
        
        """
        nb_visible = 64
        nb_hidden = 32
        nb_z = 16
        nb_epochs_training = 60000
        
        visible_units = numpy.random.normal(loc=0.,
                                            scale=5.,
                                            size=(20, nb_visible))
        variational_ae = VariationalAutoencoder(nb_visible,
                                                nb_hidden,
                                                nb_z,
                                                True,
                                                0.3)
        (scaled_kld, rec_error) = variational_ae.evaluation(visible_units)
        print('Scaled KL divergence before the training: {}'.format(scaled_kld))
        print('Reconstruction error before the training: {}'.format(rec_error))
        for _ in range(nb_epochs_training):
            variational_ae.training(visible_units)
        (scaled_kld, rec_error) = variational_ae.evaluation(visible_units)
        print('Scaled KL divergence after {0} training epochs: {1}'.format(nb_epochs_training, scaled_kld))
        print('Reconstruction error after {0} training epochs: {1}'.format(nb_epochs_training, rec_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests two methods of class `VariationalAutoencoder`.')
    parser.add_argument('name', help='name of the method to be tested')
    args = parser.parse_args()
    
    tester = TesterVariationalAutoencoder()
    getattr(tester, 'test_' + args.name)()


