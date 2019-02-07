"""A library that contains functions dedicated to JPEG2000."""

import glymur
import numpy
import os
import subprocess

import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def compress_jpeg2000(qualities, nb_images, path_to_before_jpeg2000, path_to_after_jpeg2000):
    """Compresses the luminance images via JPEG2000.
    
    For each compression quality, all the luminance
    images are compressed via JPEG2000 and the result
    of the compression is saved.
    
    Parameters
    ----------
    qualities : list
        Each integer in this list is
        a compression quality.
    nb_images : int
        Number of luminance images.
    path_to_before_jpeg2000 : str
        Path to the folder containing
        the luminance images before
        being compressed via JPEG2000.
    path_to_after_jpeg2000 : str
        Path to the folder in which the
        luminance images after being
        compressed via JPEG2000 are saved.
    
    """
    for quality in qualities:
        for i in range(nb_images):
            path_to_reference = os.path.join(path_to_before_jpeg2000,
                                             'reference_{}.png'.format(i))
            path_to_directory_reconstruction = os.path.join(path_to_after_jpeg2000,
                                                            'quality_{}'.format(quality))
            
            # The directory containing the reconstructed images
            # is created if it does not exist.
            if not os.path.exists(path_to_directory_reconstruction):
                os.makedirs(path_to_directory_reconstruction)
            path_to_reconstruction = os.path.join(path_to_directory_reconstruction,
                                                  'reconstruction_{}.jp2'.format(i))
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
            subprocess.check_call(args_subprocess,
                                  shell=False)

def compute_rate_psnr(quality, list_rotation, index, path_to_before_jpeg2000, path_to_after_jpeg2000, positions_top_left):
    """Computes the rate and the PSNR associated to the compression of the luminance image via JPEG2000.
    
    The compression of a single luminance image
    via JPEG2000 at a single compression quality
    is considered.
    
    Parameters
    ----------
    quality : int
        Compression quality.
    list_rotation : list
        Each integer in this list is the index
        of a rotated luminance image.
    index : int
        Luminance image index.
    path_to_before_jpeg2000 : str
        Path to the folder containing
        the luminance images before
        being compressed via JPEG2000.
    path_to_after_jpeg2000 : str
        Path to the folder containing
        the luminance images after being
        compressed via JPEG2000.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        This array is dedicated to visualization.
        `positions_top_left[:, i]` contains the
        row and the column of the image pixel at
        the top-left of the ith crop of the single
        luminance image after being compressed
        via JPEG2000.
    
    Returns
    -------
    tuple
        float
            Rate associated to the compression
            of the luminance image via JPEG2000.
        numpy.float64
            PSNR associated to the compression
            of the luminance image via JPEG2000.
    
    """
    path_to_reference = os.path.join(path_to_before_jpeg2000,
                                     'reference_{}.png'.format(index))
    path_to_reconstruction = os.path.join(path_to_after_jpeg2000,
                                          'quality_{}'.format(quality),
                                          'reconstruction_{}.jp2'.format(index))
    reference_uint8 = tls.read_image_mode(path_to_reference,
                                          'L')
    
    # `Glymur` is needed to read JPEG2000 images.
    file = glymur.Jp2k(path_to_reconstruction)
    reconstruction_uint8 = file[:]
    psnr = tls.psnr_2d(reference_uint8, reconstruction_uint8)
    nb_bytes = os.stat(path_to_reconstruction).st_size
    rate = float(8*nb_bytes)/numpy.prod(reference_uint8.shape)
    
    # In Python 2.x, the loop control variable of
    # list comprehensions leak into the surrounding
    # scope. It is no longer the case in Python 3.x.
    paths = [os.path.join(path_to_after_jpeg2000, 'quality_{}'.format(quality), 'reconstruction_{}.png'.format(index))]
    paths += [os.path.join(path_to_after_jpeg2000, 'quality_{}'.format(quality), 'reconstruction_{0}_crop_{1}.png'.format(index, index_crop)) for index_crop in range(positions_top_left.shape[1])]
    tls.visualize_rotated_luminance(reconstruction_uint8,
                                    index in list_rotation,
                                    positions_top_left,
                                    paths)
    return (rate, psnr)

def compute_rates_psnrs(qualities, list_rotation, nb_images, path_to_before_jpeg2000, path_to_after_jpeg2000, positions_top_left):
    """Computes a series of pairs (rate, PSNR).
    
    For each compression quality, for each
    luminance image, the rate and the PSNR
    associated to the compression of the
    luminance image via JPEG2000 are computed.
    
    Parameters
    ----------
    qualities : list
        Each integer in this list is
        a compression quality.
    list_rotation : list
        Each integer in this list is the index
        of a rotated luminance image.
    nb_images : int
        Number of luminance images.
    path_to_before_jpeg2000 : str
        Path to the folder containing
        the luminance images before
        being compressed via JPEG2000.
    path_to_after_jpeg2000 : str
        Path to the folder containing
        the luminance images after being
        compressed via JPEG2000.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        This array is dedicated to visualization.
        `positions_top_left[:, i]` contains the
        row and the column of the image pixel
        at the top-left of the ith crop of each
        luminance image after being compressed
        via JPEG2000.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the rate associated to the compression
            of the jth luminance image at the ith compression
            quality.
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the PSNR associated to the compression
            of the jth luminance image at the ith compression
            quality.
    
    """
    nb_points = len(qualities)
    rate = numpy.zeros((nb_points, nb_images))
    psnr = numpy.zeros((nb_points, nb_images))
    for i in range(nb_points):
        for j in range(nb_images):
            (rate[i, j], psnr[i, j]) = compute_rate_psnr(qualities[i],
                                                         list_rotation,
                                                         j,
                                                         path_to_before_jpeg2000,
                                                         path_to_after_jpeg2000,
                                                         positions_top_left)
    
    return (rate, psnr)

def evaluate_jpeg2000(reference_uint8, path_to_before_jpeg2000, path_to_after_jpeg2000, qualities, list_rotation, positions_top_left):
    """Evaluates JPEG2000 on the luminance images in terms of rate-distortion.
    
    The luminance images are written. Then, for
    each compression quality, all the luminance
    images are compressed via JPEG2000. Finally,
    for each compression quality, for each luminance
    image, the rate and the PSNR associated to the
    compression of the luminance image via JPEG2000
    are computed.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance images. `reference_uint8[i, :, :]`
        is the ith luminance image.
    path_to_before_jpeg2000 : str
        Path to the folder containing
        the luminance images before
        being compressed via JPEG2000.
    path_to_after_jpeg2000 : str
        Path to the folder containing
        the luminance images after being
        compressed via JPEG2000.
    qualities : list
        Each integer in this list is
        a compression quality.
    list_rotation : list
        Each integer in this list is the index
        of a rotated luminance image.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        This array is dedicated to visualization.
        `positions_top_left[:, i]` contains the
        row and the column of the image pixel
        at the top-left of the ith crop of each
        luminance image after being compressed
        via JPEG2000.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the rate associated to the compression
            of the jth luminance image at the ith compression
            quality.
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the PSNR associated to the compression
            of the jth luminance image at the ith compression
            quality.
    
    """
    write_luminances(reference_uint8,
                     path_to_before_jpeg2000)
    nb_images = reference_uint8.shape[0]
    compress_jpeg2000(qualities,
                      nb_images,
                      path_to_before_jpeg2000,
                      path_to_after_jpeg2000)
    (rate, psnr) = compute_rates_psnrs(qualities,
                                       list_rotation,
                                       nb_images,
                                       path_to_before_jpeg2000,
                                       path_to_after_jpeg2000,
                                       positions_top_left)
    return (rate, psnr)

def write_luminances(reference_uint8, path_to_before_jpeg2000):
    """Writes the luminance images.
    
    Parameters
    ----------
    reference_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance images. `reference_uint8[i, :, :]`
        is the ith luminance image.
    path_to_before_jpeg2000 : str
        Path to the folder in which
        the luminance images are saved.
    
    """
    for i in range(reference_uint8.shape[0]):
        tls.save_image(os.path.join(path_to_before_jpeg2000, 'reference_{}.png'.format(i)),
                       reference_uint8[i, :, :])


