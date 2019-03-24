"""A script to compare the errors of entropy approximation via two relations."""

import argparse
import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy.stats.distributions

import tools.tools as tls

def approximate_entropy_plot_errors(y, grid, low_projection, nb_points_per_interval, nb_intervals_per_side,
                                    nb_epochs_fitting, bin_widths, theoretical_diff_entropy,
                                    path_to_normed_hist_prefix, path_to_plot_errors):
    """Approximates the entropy of the quantized samples via two relations and plots the errors of entropy approximation.
    
    The 1st relation is derived from Theorem 8.3.1 in the
    book "Elements of information theory, second edition",
    written by Thomas M. Cover and Joy A. Thomas. The 2nd
    relation is implemented by the function `approximate_entropy`.
    
    Parameters
    ----------
    y : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples to be quantized. The entropy of the
        quantized samples will be displayed in the plot.
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear function. Thanks to
        `low_projection`, the parameters of the piecewise
        linear function cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    nb_epochs_fitting : int
        Number of fitting epochs.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Quantization bin widths.
    theoretical_diff_entropy : float
        Theoretical differential entropy of the probability
        density function of `y`.
    path_to_normed_hist_prefix : str
        Prefix of the path to the saved normed histogram
        of the noisy samples.
    path_to_plot_errors : str
        Path to the saved plot of the errors of entropy
        approximation. The path ends with ".png".
    
    """
    gaps = numpy.zeros((2, bin_widths.size))
    for i in range(bin_widths.size):
        bin_width = bin_widths[i].item()
        approx_entropy_0 = theoretical_diff_entropy - numpy.log2(bin_width)
        quantized_y = tls.quantization(y, bin_width)
        disc_entropy = tls.discrete_entropy(quantized_y,
                                            bin_width)
        gaps[0, i] = numpy.absolute(disc_entropy - approx_entropy_0)
        samples_uniform = numpy.random.uniform(low=-0.5*bin_width,
                                               high=0.5*bin_width,
                                               size=y.size)
        y_tilde = y + samples_uniform
        
        # `parameters` are the parameters of the piecewise
        # linear function. The piecewise linear function
        # approximates the probability density function of
        # `y_tilde`. Note that the probability density function
        # of `y_tilde` is the convolution between the probability
        # density function of `y` and the probability density
        # function of the continuous uniform distribution of
        # support [-0.5*`bin_width`, 0.5*`bin_width`].
        parameters = fit_piecewise_linear_function(y_tilde,
                                                   grid,
                                                   low_projection,
                                                   nb_points_per_interval,
                                                   nb_intervals_per_side,
                                                   nb_epochs_fitting)
        
        # The normed histogram of the noisy samples is
        # saved to check the reliability of the fitting.
        if i == 0 or i == bin_widths.size - 1:
            tls.normed_histogram(y_tilde,
                                 grid,
                                 parameters,
                                 'Noisy samples after the fitting when $\delta$ = {}'.format(bin_width),
                                 path_to_normed_hist_prefix + '_' + tls.float_to_str(bin_width) + '.png')
        approx_entropy_1 = tls.approximate_entropy(y_tilde,
                                                   parameters,
                                                   nb_points_per_interval,
                                                   nb_intervals_per_side,
                                                   bin_width)
        gaps[1, i] = numpy.absolute(disc_entropy - approx_entropy_1)
    plot_errors(bin_widths,
                gaps,
                path_to_plot_errors)

def compare_errors_pdfs(grid, low_projection, nb_points_per_interval, nb_intervals_per_side,
                        nb_epochs_fitting, bin_widths, tuples_pdf_scale, theoretical_diff_entropies,
                        nb_samples, paths_to_normed_hist_prefix, paths_to_plot_errors):
    """Compares the errors of entropy approximation via the two relations for different probability density functions.
    
    Parameters
    ----------
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear function. Thanks to
        `low_projection`, the parameters of the piecewise
        linear function cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    nb_epochs_fitting : int
        Number of fitting epochs.
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Quantization bin widths.
    tuples_pdf_scale : tuple
        Each tuple in this tuple defines a continuous
        probability distribution via its family and its
        scale.
    theoretical_diff_entropies : list
        `theoretical_diff_entropies[i]` is the differential
        entropy of the continuous probability distribution
        defined by `tuples_pdf_scale[i]`.
    nb_samples : int
        Number of samples from each continuous probability
        distribution.
    paths_to_normed_hist_prefix : list
        Each string in this list is the prefix of the path
        to a saved normed histogram of noisy samples.
    paths_to_plot_errors : list
        Each string in this list is the path to a saved
        plot of errors of entropy approximation. Each path
        ends with ".png".
    
    Raises
    ------
    ValueError
        If `len(theoretical_diff_entropies)` is not equal to `len(tuples_pdf_scale)`.
    ValueError
        If `len(paths_to_normed_hist_prefix)` is not equal to `len(tuples_pdf_scale)`.
    ValueError
        If `len(paths_to_plot_errors)` is not equal to `len(tuples_pdf_scale)`.
    
    """
    nb_pdfs = len(tuples_pdf_scale)
    if len(theoretical_diff_entropies) != nb_pdfs:
        raise ValueError('`len(theoretical_diff_entropies)` is not equal to `len(tuples_pdf_scale)`.')
    if len(paths_to_normed_hist_prefix) != nb_pdfs:
        raise ValueError('`len(paths_to_normed_hist_prefix)` is not equal to `len(tuples_pdf_scale)`.')
    if len(paths_to_plot_errors) != nb_pdfs:
        raise ValueError('`len(paths_to_plot_errors)` is not equal to `len(tuples_pdf_scale)`.')
    for i in range(nb_pdfs):
        y = tuples_pdf_scale[i][0](loc=0.,
                                   scale=tuples_pdf_scale[i][1],
                                   size=nb_samples)
        approximate_entropy_plot_errors(y,
                                        grid,
                                        low_projection,
                                        nb_points_per_interval,
                                        nb_intervals_per_side,
                                        nb_epochs_fitting,
                                        bin_widths,
                                        theoretical_diff_entropies[i],
                                        paths_to_normed_hist_prefix[i],
                                        paths_to_plot_errors[i])

def compute_diff_entropy_laplace(scale):
    """Computes the differential entropy of the Laplace distribution.
    
    The difference between `compute_diff_entropy_laplace` and `scipy.stats.laplace.entropy`
    is a factor `numpy.log(2.)`.
    
    Parameters
    ----------
    scale : float
        Scale of the Laplace distribution.
    
    Returns
    -------
    float
        Differential entropy of the Laplace distribution.
    
    """
    return ((1. + numpy.log(2.*scale))/numpy.log(2.)).item()

def compute_diff_entropy_logistic(scale):
    """Computes the differential entropy of the logistic distribution.
    
    The difference between `compute_diff_entropy_logistic` and `scipy.stats.logistic.entropy`
    is a factor `numpy.log(2.)`.
    
    Parameters
    ----------
    scale : float
        Scale of the logistic distribution.
    
    Returns
    -------
    float
        Differential entropy of the logistic distribution.
    
    """
    return ((2. + numpy.log(scale))/numpy.log(2.)).item()

def compute_diff_entropy_normal(scale):
    """Computes the differential entropy of the normal distribution.
    
    The difference between `compute_diff_entropy_normal` and `scipy.stats.norm.entropy`
    is a factor `numpy.log(2.)`.
    
    Parameters
    ----------
    scale : float
        Scale of the normal distribution.
    
    Returns
    -------
    float
        Differential entropy of the normal distribution.
    
    """
    return (0.5*(1. + numpy.log(2.*numpy.pi*scale**2))/numpy.log(2.)).item()

def fit_piecewise_linear_function(samples, grid, low_projection, nb_points_per_interval,
                                  nb_intervals_per_side, nb_epochs_fitting, learning_rate=0.2):
    """Fits a piecewise linear function to the unknown probability density function.
    
    Parameters
    ----------
    samples : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Samples from the unknown probability density
        function.
    grid : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Grid storing the sampling points.
    low_projection : float
        Strictly positive minimum for the parameters
        of the piecewise linear function. Thanks to
        `low_projection`, the parameters of the piecewise
        linear function cannot get extremely close to 0.
        Therefore, the limited floating-point precision
        cannot round them to 0.
    nb_points_per_interval : int
        Number of sampling points per unit interval
        in the grid.
    nb_intervals_per_side : int
        Number of unit intervals in the right half
        of the grid. The grid is symmetrical about 0.
    nb_epochs_fitting : int
        Number of fitting epochs.
    learning_rate : float, optional
        Learning rate for the parameters of the piecewise
        linear function. The default value is 0.2.
    
    Returns
    -------
    numpy.ndarray
        1D array with data-type `numpy.float64`.
        Parameters of the piecewise linear function
        after the fitting.
    
    """
    parameters = numpy.maximum(scipy.stats.distributions.cauchy.pdf(grid),
                               low_projection)
    for _ in range(nb_epochs_fitting):
        gradients = tls.gradient_density_approximation(samples,
                                                       parameters,
                                                       nb_points_per_interval,
                                                       nb_intervals_per_side)
        parameters -= learning_rate*gradients
        parameters = numpy.maximum(parameters,
                                   low_projection)
    return parameters

def plot_errors(bin_widths, gaps, path):
    """Plots the errors of entropy approximation for the two relations.
    
    Parameters
    ----------
    bin_widths : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Quantization bin widths.
    gaps : numpy.ndarray
        2D array with data-type `numpy.float64`.
        Errors of entropy approximation for each of the
        two relations. `gaps[0, :]` stores the errors of
        entropy approximation for the 1st relation. `gaps[1, :]`
        stores the errors of entropy approximation for the
        2nd relation.
    path : str
        Path to the saved plot of the errors of entropy
        approximation. The path ends with ".png".
    
    Raises
    ------
    ValueError
        If `bin_widths.size` is not equal to 1.
    ValueError
        If `gaps.shape[0]` is not equal to 2.
    ValueError
        If `bin_widths.size` is not equal to `gaps.shape[1]`.
    
    """
    if bin_widths.ndim != 1:
        raise ValueError('`bin_widths.size` is not equal to 1.')
    
    # If `gaps.ndim` is not equal to 2, the
    # unpacking below raises a `ValueError`.
    (nb_graphs, nb_gap_values) = gaps.shape
    if nb_graphs != 2:
        raise ValueError('`gaps.shape[0]` is not equal to 2.')
    if bin_widths.size != nb_gap_values:
        raise ValueError('`bin_widths.size` is not equal to `gaps.shape[1]`.')
    handle = []
    handle.append(plt.plot(bin_widths,
                           gaps[0, :],
                           color='blue')[0])
    handle.append(plt.plot(bin_widths,
                           gaps[1, :],
                           color='red')[0])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(r'Evolution of the two errors with $\delta$',
              fontsize=20)
    plt.xlabel(r'$\delta$',
               fontsize=20)
    plt.ylabel('error of entropy approximation',
               fontsize=20)
    plt.legend(handle,
               [r'A[1]', r'A[2]'],
               loc='upper center',
               prop={'size':20},
               frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compares the errors of entropy approximation via two relations.')
    parser.parse_args()
    
    nb_points_per_interval = 4
    nb_intervals_per_side = 50
    bin_widths = numpy.linspace(0.2, 6., num=30)
    low_projection = 1.e-6
    nb_epochs_fitting = 120
    nb_samples = 200000
        
    # `tuples_pdf_scale[i]` is the tuple containing
    # the probability density function of index i and
    # its scale.
    tuples_pdf_scale = (
        (numpy.random.normal, 0.5),
        (numpy.random.normal, 1.),
        (numpy.random.normal, 2.),
        (numpy.random.logistic, 1.),
        (numpy.random.logistic, 2.),
        (numpy.random.laplace, 0.5),
        (numpy.random.laplace, 1.),
        (numpy.random.laplace, 2.)
    )
    
    # `theoretical_diff_entropies[i]` is the differential
    # entropy of the probability density function `tuples_pdf_scale[i][0]`
    # with scale `tuples_pdf_scale[i][1]`.
    theoretical_diff_entropies = [compute_diff_entropy_normal(tuples_pdf_scale[i][1]) for i in range(3)]
    theoretical_diff_entropies += [compute_diff_entropy_logistic(tuples_pdf_scale[3 + i][1]) for i in range(2)]
    theoretical_diff_entropies += [compute_diff_entropy_laplace(tuples_pdf_scale[5 + i][1]) for i in range(3)]
    paths_to_normed_hist_prefix = [
        'supplementary/normed_histogram_normal_{}'.format(tls.float_to_str(tuples_pdf_scale[i][1])) for i in range(3)
    ]
    paths_to_normed_hist_prefix += [
        'supplementary/normed_histogram_logistic_{}'.format(tls.float_to_str(tuples_pdf_scale[3 + i][1])) for i in range(2)
    ]
    paths_to_normed_hist_prefix += [
        'supplementary/normed_histogram_laplace_{}'.format(tls.float_to_str(tuples_pdf_scale[5 + i][1])) for i in range(3)
    ]
    paths_to_plot_errors = [
        'supplementary/approx_entropy_normal_{}.png'.format(tls.float_to_str(tuples_pdf_scale[i][1])) for i in range(3)
    ]
    paths_to_plot_errors += [
        'supplementary/approx_entropy_logistic_{}.png'.format(tls.float_to_str(tuples_pdf_scale[3 + i][1])) for i in range(2)
    ]
    paths_to_plot_errors += [
        'supplementary/approx_entropy_laplace_{}.png'.format(tls.float_to_str(tuples_pdf_scale[5 + i][1])) for i in range(3)
    ]
    
    nb_points = 2*nb_points_per_interval*nb_intervals_per_side + 1
    grid = numpy.linspace(-nb_intervals_per_side,
                          nb_intervals_per_side,
                          num=nb_points)
    
    # In the function `compare_errors_pdfs`, the samples
    # from a given continuous probability distribution are
    # destroyed before taking a new continuous probability
    # distribution and sampling from it.
    compare_errors_pdfs(grid,
                        low_projection,
                        nb_points_per_interval,
                        nb_intervals_per_side,
                        nb_epochs_fitting,
                        bin_widths,
                        tuples_pdf_scale,
                        theoretical_diff_entropies,
                        nb_samples,
                        paths_to_normed_hist_prefix,
                        paths_to_plot_errors)
    


