"""A library that contains functions dedicated to JPEG and JPEG2000."""

import glymur
import numpy
import os
import scipy.misc
import subprocess

import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def compress_jpeg(qualities, nb_images, path_to_before, path_to_after, is_2000):
    """Compresses the RGB images via either JPEG or JPEG2000.
    
    For each compression quality, all the RGB
    images are compressed via either JPEG or
    JPEG2000 and the result of the compression
    is saved.
    
    Parameters
    ----------
    qualities : list
        Each integer in this list is
        a compression quality.
    nb_images : int
        Number of RGB images.
    path_to_before : str
        Path to the folder containing
        the RGB images before being compressed
        via either JPEG or JPEG2000.
    path_to_after : str
        Path to the folder in which the
        RGB images after being compressed
        via either JPEG or JPEG2000 are saved.
    is_2000 : bool
        Is it JPEG2000?
    
    """
    if is_2000:
        tag = 'jpeg2000'
        extension = 'jp2'
    else:
        tag = 'jpeg'
        extension = 'jpg'
    for quality in qualities:
        for i in range(nb_images):
            path_to_reference = os.path.join(path_to_before,
                                             'reference_{}.png'.format(i))
            path_to_reconstruction = os.path.join(path_to_after,
                                                  tag,
                                                  'quality_{}'.format(quality),
                                                  'reconstruction_{0}.{1}'.format(i, extension))
            args_subprocess = [
                'magick',
                'convert',
                path_to_reference,
                '-quality',
                str(quality),
                path_to_reconstruction
            ]
            
            # Setting `shell` to True makes the program
            # vulnerable to shell injection, see
            # <https://docs.python.org/2/library/subprocess.html>.
            subprocess.call(args_subprocess, shell=False)

def compute_rate_psnr(path_to_reference, path_to_reconstruction, is_2000):
    """Computes the rate and the PSNR associated to the compression of the RGB image via either JPEG or JPEG2000.
    
    The compression of a single RGB image
    via either JPEG or JPEG2000 at a single
    compression quality is considered.
    
    Parameters
    ----------
    path_to_reference : str
        Path to the RGB image before
        being compressed via either
        JPEG or JPEG2000.
    path_to_reconstruction : str
        Path to the RGB image after
        being compressed via either
        JPEG or JPEG2000. If JPEG is
        used, `path_to_reconstruction`
        must end with ".jpg". Otherwise,
        it must end with ".jp2".
    is_2000 : bool
        Is it JPEG2000?
    
    Returns
    -------
    tuple
        numpy.float64
            Rate associated to the compression of
            the RGB image via either JPEG or JPEG2000.
        numpy.float64
            PSNR associated to the compression of
            the RGB image via either JPEG or JPEG2000.
    
    """
    reference_uint8 = scipy.misc.imread(path_to_reference)
    reference_float64 = reference_uint8.astype(numpy.float64)
    
    # `Glymur` is needed to read JPEG2000 images.
    if is_2000:
        file = glymur.Jp2k(path_to_reconstruction)
        reconstruction_uint8 = file[:]
    else:
        reconstruction_uint8 = scipy.misc.imread(path_to_reconstruction)
    reconstruction_float64 = reconstruction_uint8.astype(numpy.float64)
    psnr = 10.*numpy.log10((255.**2)/numpy.mean((reference_float64 - reconstruction_float64)**2))
    nb_bytes = os.stat(path_to_reconstruction).st_size
    rate = float(8*nb_bytes)/numpy.prod(reference_uint8.shape)
    return (rate, psnr)

def compute_rates_psnrs(qualities, nb_images, path_to_before, path_to_after, is_2000):
    """Computes a series of pairs (rate, PSNR).
    
    For each compression quality, for each
    RGB image, the rate and the PSNR associated
    to the compression of the RGB image via either
    JPEG or JPEG2000 are computed.
    
    Parameters
    ----------
    qualities : list
        Each integer in this list is
        a compression quality.
    nb_images : int
        Number of RGB images.
    path_to_before : str
        Path to the folder containing
        the RGB images before being compressed
        via either JPEG or JPEG2000.
    path_to_after : str
        Path to the folder containing the
        RGB images after being compressed
        via either JPEG or JPEG2000.
    is_2000 : bool
        Is it JPEG2000?
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the rate associated to the compression
            of the jth RGB image at the ith compression
            quality.
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the PSNR associated to the compression
            of the jth RGB image at the ith compression
            quality.
    
    """
    if is_2000:
        tag = 'jpeg2000'
        extension = 'jp2'
    else:
        tag = 'jpeg'
        extension = 'jpg'
    nb_points = len(qualities)
    rate = numpy.zeros((nb_points, nb_images))
    psnr = numpy.zeros((nb_points, nb_images))
    for i in range(nb_points):
        for j in range(nb_images):
            path_to_reference = os.path.join(path_to_before,
                                             'reference_{}.png'.format(j))
            path_to_directory_reconstruction = os.path.join(path_to_after,
                                                            tag,
                                                            'quality_{}'.format(qualities[i]))
            
            # The directory containing the reconstructed images
            # is created if it does not exist.
            if not os.path.exists(path_to_directory_reconstruction):
                os.makedirs(path_to_directory_reconstruction)
            path_to_reconstruction = os.path.join(path_to_directory_reconstruction,
                                                  'reconstruction_{0}.{1}'.format(j, extension))
            (rate[i, j], psnr[i, j]) = compute_rate_psnr(path_to_reference,
                                                         path_to_reconstruction,
                                                         is_2000)
    return (rate, psnr)

def evaluate_jpeg(reference_uint8, path_to_before, path_to_after, qualities_jpeg, qualities_jpeg2000):
    """Evaluates JPEG and JPEG2000 on the RGB digits in terms of rate-distortion.
    
    The RGB digits are written. Then, for each
    compression quality, all the RGB digits are
    compressed via JPEG. For each compression
    quality, for each RGB digit, the rate and
    the PSNR associated to the compression of
    the RGB digit via JPEG are computed. Finally,
    for each compression quality, the rate and
    the PSNR are averaged over all RGB digits.
    The same steps are repeated using JPEG2000
    instead of JPEG.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        RGB digits. `reference_uint8[i, :]`
        contains the ith RGB digit.
    path_to_before : str
        Path to the folder containing
        the RGB images before being compressed
        via either JPEG or JPEG2000.
    path_to_after : str
        Path to the folder containing the
        RGB images after being compressed
        via either JPEG or JPEG2000.
    qualities_jpeg : list
        JPEG compression qualities.
    qualities_jpeg2000 : list
        JPEG2000 compression qualities.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the JPEG rate averaged
            over all RGB digits at the ith compression
            quality.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the JPEG PSNR averaged
            over all RGB digits at the ith compression
            quality.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the JPEG2000 rate
            averaged over all RGB digits at the
            ith compression quality.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the JPEG2000 PSNR
            averaged over all RGB digits at the
            ith compression quality.
    
    """
    write_digits(reference_uint8, path_to_before)
    nb_images = reference_uint8.shape[0]
    compress_jpeg(qualities_jpeg,
                  nb_images,
                  path_to_before,
                  path_to_after,
                  False)
    compress_jpeg(qualities_jpeg2000,
                  nb_images,
                  path_to_before,
                  path_to_after,
                  True)
    
    # The prefix "idv" means "individual". The
    # rate has not been averaged over all RGB
    # digits yet.
    (rate_jpeg_idv, psnr_jpeg_idv) = compute_rates_psnrs(qualities_jpeg,
                                                         nb_images,
                                                         path_to_before,
                                                         path_to_after,
                                                         False)
    (rate_jpeg2000_idv, psnr_jpeg2000_idv) = compute_rates_psnrs(qualities_jpeg2000,
                                                                 nb_images,
                                                                 path_to_before,
                                                                 path_to_after,
                                                                 True)
    rate_jpeg = numpy.mean(rate_jpeg_idv, axis=1)
    psnr_jpeg = numpy.mean(psnr_jpeg_idv, axis=1)
    rate_jpeg2000 = numpy.mean(rate_jpeg2000_idv, axis=1)
    psnr_jpeg2000 = numpy.mean(psnr_jpeg2000_idv, axis=1)
    return (rate_jpeg, psnr_jpeg, rate_jpeg2000, psnr_jpeg2000)

def write_digits(reference_uint8, path_to_before):
    """Writes the RGB digits.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        RGB digits. `reference_uint8[i, :]`
        contains the ith RGB digit.
    path_to_before : str
        Path to the folder in which the
        RGB digits are saved.
    
    """
    # The function `tls.row_to_images` checks that
    # `reference_uint8.dtype` is equal to `numpy.uint8`
    # and `reference_uint8.ndim` is equal to 2.
    images_uint8 = tls.rows_to_images(reference_uint8, 32, 32)
    for i in range(images_uint8.shape[3]):
        scipy.misc.imsave(os.path.join(path_to_before, 'reference_{}.png'.format(i)),
                          images_uint8[:, :, :, i])


