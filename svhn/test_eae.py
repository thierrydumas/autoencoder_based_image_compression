"""A script to test five methods of class `EntropyAutoencoder`."""

import argparse
import numpy

from eae.EntropyAutoencoder import EntropyAutoencoder


class TesterEntropyAutoencoder(object):
    """Class for testing five methods of class `EntropyAutoencoder`."""
    
    def test_backpropagation_eae_bw(self):
        """Tests the method `backproagation_eae_bw`.
        
        Five histograms are saved in the folder
        "eae/pseudo_visualization/backpropagation/".
        The test is successful if, in each
        histogram, the absolute values are
        smaller than 1.e-9.
        
        """
        nb_visible = 20
        nb_hidden = 15
        nb_y = 12
        
        # Each visible unit is modeled as
        # continuous random variable with
        # Gaussian probability density function.
        visible_units = numpy.random.normal(loc=0.,
                                            scale=1.,
                                            size=(10, nb_visible))
        entropy_ae = EntropyAutoencoder(nb_visible,
                                        nb_hidden,
                                        nb_y,
                                        1.5,
                                        0.75,
                                        False)
        entropy_ae.backpropagation_eae_bw(visible_units,
                                          is_checking=True,
                                          path_to_checking_g='eae/pseudo_visualization/backpropagation/')
    
    def test_backpropagation_fct(self):
        """Tests the method `backpropagation_fct`.
        
        The method `backpropagation_fct` computes
        the gradients of the parameters of the piecewise
        linear function. It is tested via gradient
        checking, see
        <http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization>.
        
        A histogram is saved at
        "eae/pseudo_visualization/backpropagation/gfct.png".
        The test is successful if the histogram
        absolute values are smaller than 1.e-9.
        
        """
        nb_visible = 20
        nb_hidden = 15
        nb_y = 12
        
        # Each visible unit is modeled as
        # continuous random variable with
        # Gaussian probability density function.
        visible_units = numpy.random.normal(loc=0.,
                                            scale=1.,
                                            size=(10, nb_visible))
        entropy_ae = EntropyAutoencoder(nb_visible,
                                        nb_hidden,
                                        nb_y,
                                        1.5,
                                        0.75,
                                        False)
        entropy_ae.backpropagation_fct(visible_units,
                                       is_checking=True,
                                       path_to_checking_g='eae/pseudo_visualization/backpropagation/')
    
    def test_training_eae_bw(self):
        """Tests the method `training_eae_bw`.
        
        The test is successful if the
        reconstruction error after the training
        is much smaller than the reconstruction
        error before the training.
        
        """
        nb_visible = 64
        nb_hidden = 32
        nb_y = 16
        nb_epochs_training = 10000
        
        visible_units = numpy.random.normal(loc=0.,
                                            scale=5.,
                                            size=(20, nb_visible))
        entropy_ae = EntropyAutoencoder(nb_visible,
                                        nb_hidden,
                                        nb_y,
                                        1.5,
                                        0.75,
                                        False)
        
        # Here, the number of unit intervals in the
        # right half of the grid before the training
        # can be smaller than the number of unit intervals
        # in the right half of the grid after the training.
        (_, _, scaled_approx_entropy_0, rec_error_0, _, _) = entropy_ae.evaluation(visible_units)
        print('Number of unit intervals in the right half of the grid before the training: {}'.format(entropy_ae.nb_intervals_per_side))
        print('Scaled approximate entropy before the training: {}'.format(scaled_approx_entropy_0))
        print('Reconstruction error before the training: {}'.format(rec_error_0))
        for _ in range(nb_epochs_training):
            entropy_ae.training_fct(visible_units)
            entropy_ae.training_eae_bw(visible_units)
        (_, _, scaled_approx_entropy_1, rec_error_1, _, _) = entropy_ae.evaluation(visible_units)
        print('Number of unit intervals in the right half of the grid after {0} training epochs: {1}'.format(nb_epochs_training, entropy_ae.nb_intervals_per_side))
        print('Scaled approximate entropy after {0} training epochs: {1}'.format(nb_epochs_training, scaled_approx_entropy_1))
        print('Reconstruction error after {0} training epochs: {1}'.format(nb_epochs_training, rec_error_1))
    
    def test_training_fct(self):
        """Tests the method `training_fct`.
        
        The test is successful if the loss of the
        approximation of the probability density
        function of the latent variables perturbed
        by uniform noise with the piecewise linear
        function after the fitting is smaller than
        the one before the fitting.
        
        """
        nb_visible = 64
        nb_hidden = 32
        nb_y = 16
        nb_epochs_fitting = 1000
        
        visible_units = numpy.random.normal(loc=0.,
                                            scale=5.,
                                            size=(20, nb_visible))
        entropy_ae = EntropyAutoencoder(nb_visible,
                                        nb_hidden,
                                        nb_y,
                                        1.5,
                                        0.75,
                                        False)
        
        # Here, the number of unit intervals in the right
        # half of the grid before the fitting has to be
        # equal to the number of unit intervals in the
        # right half of the grid after the fitting.
        loss_density_approx_0 = entropy_ae.evaluation(visible_units)[4]
        print('Number of unit intervals in the right half of the grid before the fitting: {}'.format(entropy_ae.nb_intervals_per_side))
        print('Loss of the approximation of the probability density function before the fitting: {}'.format(loss_density_approx_0))
        for _ in range(nb_epochs_fitting):
            entropy_ae.training_fct(visible_units)
        loss_density_approx_1 = entropy_ae.evaluation(visible_units)[4]
        print('Number of unit intervals in the right half of the grid after {0} fitting epochs: {1}'.format(nb_epochs_fitting, entropy_ae.nb_intervals_per_side))
        print('Loss of the approximation of the probability density function after {0} fitting epochs: {1}'.format(nb_epochs_fitting, loss_density_approx_1))
    
    def test_weights_decay(self):
        """Tests the method `weights_decay`.
        
        The test is successful if the l2-norm
        weight decay computed by the function is
        close to the coarse l2-norm weight decay
        computed by hand.
        
        """
        nb_visible = 480
        nb_hidden = 340
        nb_y = 340
        
        entropy_ae = EntropyAutoencoder(nb_visible,
                                        nb_hidden,
                                        nb_y,
                                        1.5,
                                        0.75,
                                        False,
                                        weights_decay_p=1.)
        w_decay = entropy_ae.weights_decay()
        print('Weight decay parameter: 1.0')
        print('L2-norm weight decay computed by the function: {}'.format(w_decay))
        print('Coarse l2-norm weight decay computed by hand: {}'.format(305.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests five methods of class `EntropyAutoencoder`.')
    parser.add_argument('name', help='name of the method to be tested')
    args = parser.parse_args()
    tester = TesterEntropyAutoencoder()
    getattr(tester, 'test_' + args.name)()


