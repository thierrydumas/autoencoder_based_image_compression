"""A library that contains functions for analyzing the trained entropy autoencoders."""

import matplotlib
try:
    import PyQt5
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy.stats

def fit_latent_variables(reference_float64, entropy_ae, title, path):
    """Fits a Laplace density to the normed histogram of the latent variables in the trained entropy autoencoder.
    
    Parameters
    ----------
    reference_float64 : numpy.ndarray
        2D array with data-type `numpy.float64`.
        RGB digits after the preprocessing.
        `reference_float64[i, :]` contains the
        ith RGB digit after the preprocessing.
    entropy_ae : EntropyAutoencoder
        Entropy autoencoder trained with a
        specific scaling coefficient.
    title : str
        Title of the saved normed histogram.
    path : str
        Path to the saved normed histogram. The
        path must end with ".png".
    
    """
    y = entropy_ae.encoder(reference_float64)[1]
    edge_left = numpy.floor(numpy.amin(y)).item()
    edge_right = numpy.ceil(numpy.amax(y)).item()
    
    # The grid below contains 50 points
    # per unit interval.
    grid = numpy.linspace(edge_left,
                          edge_right,
                          num=50*int(edge_right - edge_left) + 1)
    
    # Let's assume that `y` contains i.i.d samples from
    # an unknown probability density function. The two
    # equations below result from the minimization of
    # the Kullback-Lieber divergence of the unknown
    # probability density function from our statistical
    # model (Laplace density of location `laplace_location`
    # and scale `laplace_scale`). Note that this minimization
    # is equivalent to the maximum likelihood estimator.
    # To dive into the details, see:
    # "Estimating distributions and densities". 36-402,
    # advanced data analysis, CMU, 27 January 2011.
    laplace_location = numpy.mean(y).item()
    laplace_scale = numpy.mean(numpy.absolute(y - laplace_location)).item()
    laplace_pdf = scipy.stats.laplace.pdf(grid,
                                          loc=laplace_location,
                                          scale=laplace_scale)
    handle = [plt.plot(grid, laplace_pdf, color='red')[0]]
    hist, bin_edges = numpy.histogram(y,
                                      bins=60,
                                      density=True)
    plt.bar(bin_edges[0:60],
            hist,
            width=bin_edges[1] - bin_edges[0],
            align='edge',
            color='blue')
    plt.title(title)
    plt.legend(handle,
               [r'$f( . ; {0}, {1})$'.format(str(round(laplace_location, 2)), str(round(laplace_scale, 2)))],
               prop={'size': 30},
               loc=9)
    plt.savefig(path)
    plt.clf()


