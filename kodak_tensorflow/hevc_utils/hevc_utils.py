"""A library that contains HEVC utilities."""

import numpy
import os
import subprocess
import sys
import warnings

import tools.tools as tls

if sys.platform.startswith('linux'):
    PATH_TO_EXE = 'HM-16.15/bin/TAppEncoderStatic'
elif sys.platform in ('win32', 'cygwin'):
    PATH_TO_EXE = 'HM-16.15/bin/vc2015/Win32/Release/TAppEncoder.exe'
else:
    PATH_TO_EXE = 'HM-16.15/bin/TAppEncoderStatic'
    warnings.warn('The OS is neither Windows nor Linux. If the HEVC executable is not at "{}", change `PATH_TO_EXE`.'.format(PATH_TO_EXE))

# The functions are sorted in
# alphabetic order.

def compress_hevc(luminance_before_hevc_uint8, path_to_before_hevc, path_to_after_hevc,
                  path_to_cfg, path_to_bitstream, qp, is_bitstream_cleaned):
    """Compresses the luminance image/video via HEVC.
    
    The luminance image/video is written in 4:0:0
    to a 1st file. Then, HEVC extracts the luminance
    image/video from the 1st file, compresses it and
    puts the luminance image/video resulting from the
    compression into a 2nd file.
    
    Parameters
    ----------
    luminance_before_hevc_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance image/video before it is
        compressed via HEVC. If `luminance_before_hevc_uint8.shape[2]`
        is equal to 1, `luminance_before_hevc_uint8`
        is viewed as a luminance image. If
        `luminance_before_hevc_uint8.shape[2]` is
        strictly larger than 1, `luminance_before_hevc_uint8`
        is viewed as a luminance video.
    path_to_before_hevc : str
        Path to the 1st file which stores the
        luminance image/video before it is compressed
        via HEVC. The path must end with ".yuv".
    path_to_after_hevc : str
        Path to the 2nd file which stores the
        luminance image/video after it is compressed
        via HEVC. The path must end with ".yuv".
    path_to_cfg : str
        Path to the configuration file. The path
        must end with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. The path
        must end with ".bin".
    qp : int
        Quantization parameter.
    is_bitstream_cleaned : bool
        Is the bistream cleaned?
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance image/video after it is
        compressed via HEVC.
    
    Raises
    ------
    TypeError
        If `luminance_before_hevc_uint8.dtype` is not
        equal to `numpy.uint8`.
    
    """
    if luminance_before_hevc_uint8.dtype != numpy.uint8:
        raise TypeError('`luminance_before_hevc_uint8.dtype` is not equal to `numpy.uint8`.')
    
    # The function `write_400` ensures that
    # `luminance_before_hevc_uint8.ndim` is
    # equal to 3.
    write_400(luminance_before_hevc_uint8, path_to_before_hevc)
    (height, width, nb_frames) = luminance_before_hevc_uint8.shape
    args_subprocess = [
        PATH_TO_EXE,
        '-c',
        path_to_cfg,
        '-i',
        path_to_before_hevc,
        '-b',
        path_to_bitstream,
        '-o',
        path_to_after_hevc,
        '-wdt',
        str(width),
        '-hgt',
        str(height),
        '--InputBitDepth=8',
        '--InputChromaFormat=400',
        '--FramesToBeEncoded={}'.format(nb_frames),
        '--QP={}'.format(qp)
    ]
    
    # Setting `shell` to True makes the program
    # vulnerable to shell injection, see
    # <https://docs.python.org/2/library/subprocess.html>.
    subprocess.check_call(args_subprocess,
                          shell=False)
    luminance_after_hevc_uint8 = read_400(path_to_after_hevc,
                                          height,
                                          width,
                                          nb_frames,
                                          numpy.uint8)
    if is_bitstream_cleaned:
        os.remove(path_to_bitstream)
    
    # The files located at respectively
    # `path_to_before_hevc` and `path_to_after_hevc`
    # are removed.
    os.remove(path_to_before_hevc)
    os.remove(path_to_after_hevc)
    return luminance_after_hevc_uint8

def compute_rate_psnr(luminances_uint8, path_to_before_hevc, path_to_after_hevc, path_to_cfg,
                      path_to_bitstream, qp, path_to_storage, list_rotation, positions_top_left):
    """Computes the rate and the PSNR associated to the compression of each luminance image via HEVC.
    
    The compression of each luminance image via
    HEVC at a single quantization parameter is
    considered. Each luminance image after it
    is compressed via HEVC is saved.
    
    Parameters
    ----------
    luminances_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance images before they are
        compressed via HEVC. `luminances_uint8[i, :, :]`
        is the ith luminance image.
    path_to_before_hevc : str
        Path to the file storing the luminance
        image before it is compressed via HEVC.
        The path must end with ".yuv".
    path_to_after_hevc : str
        Path to the file storing the luminance
        image after it is compressed via HEVC.
        The path must end with ".yuv".
    path_to_cfg : str
        Path to the configuration file. The path
        must end with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. The path
        must end with ".bin".
    qp : int
        Quantization parameter.
    path_to_storage : str
        Path to the folder in which the luminance
        images after they are compressed via HEVC
        are saved.
    list_rotation : list
        Each integer in this list is the index
        of a rotated luminance image.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        This array is dedicated to visualization.
        `positions_top_left[:, i]` contains the
        row and the column of the image pixel at
        the top-left of the ith crop of each
        luminance image after it is compressed
        via HEVC.
    
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the rate associated to
            the compression of the ith luminance image
            via HEVC.
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Its ith element is the PSNR associated to
            the compression of the ith luminance image
            via HEVC.
    
    """
    # If `luminances_uint8.ndim` is not equal to 3,
    # the unpacking below raises a `ValueError` exception.
    (nb_images, height, width) = luminances_uint8.shape
    rate = numpy.zeros(nb_images)
    psnr = numpy.zeros(nb_images)
    for i in range(nb_images):
        luminance_uint8 = luminances_uint8[i, :, :]
    
        # The function `compress_hevc` ensures that
        # `luminances_uint8.dtype` is equal to `numpy.uint8`.
        luminance_after_hevc_uint8 = compress_hevc(numpy.expand_dims(luminance_uint8, axis=2),
                                                   path_to_before_hevc,
                                                   path_to_after_hevc,
                                                   path_to_cfg,
                                                   path_to_bitstream,
                                                   qp,
                                                   False)
        reconstruction_uint8 = numpy.squeeze(luminance_after_hevc_uint8, axis=2)
        psnr[i] = tls.psnr_2d(luminance_uint8, reconstruction_uint8)
        nb_bytes = os.stat(path_to_bitstream).st_size
        rate[i] = float(8*nb_bytes)/(height*width)
        os.remove(path_to_bitstream)
        
        paths = [os.path.join(path_to_storage, 'reconstruction_{}.png'.format(i))]
        paths += [os.path.join(path_to_storage, 'reconstruction_{0}_crop_{1}.png'.format(i, index_crop)) for index_crop in range(positions_top_left.shape[1])]
        tls.visualize_rotated_luminance(reconstruction_uint8,
                                        i in list_rotation,
                                        positions_top_left,
                                        paths)
    return (rate, psnr)

def evaluate_hevc(luminances_uint8, path_to_before_hevc, path_to_after_hevc, path_to_cfg,
                  path_to_bitstream, qps, path_to_hevc_vis, list_rotation, positions_top_left):
    """Evaluates HEVC on the luminance images in terms of rate-distortion.
    
    For each quantization parameter, for each
    luminance image, the rate and the PSNR associated
    to the compression of the luminance image via HEVC
    are computed.
    
    Parameters
    ----------
    luminances_uint8 : numpy.ndarray
        3D array with data-type `numpy.uint8`.
        Luminance images before they are
        compressed via HEVC. `luminances_uint8[i, :, :]`
        is the ith luminance image.
    path_to_before_hevc : str
        Path to the file storing a luminance
        image before it is compressed via HEVC.
        The path must end with ".yuv".
    path_to_after_hevc : str
        Path to the file storing a luminance
        image after it is compressed via HEVC.
        The path must end with ".yuv".
    path_to_cfg : str
        Path to the configuration file. The path
        must end with ".cfg".
    path_to_bitstream : str
        Path to the bitstream file. The path
        must end with ".bin".
    qps : numpy.ndarray
        1D array with data-type `numpy.int32`.
        Quantization parameters.
    path_to_hevc_vis : str
        Path to the folder in which the luminance
        images after they are compressed via HEVC
        are saved.
    list_rotation : list
        Each integer in this list is the index
        of a rotated luminance image.
    positions_top_left : numpy.ndarray
        2D array with data-type `numpy.int32`.
        This array is dedicated to visualization.
        `positions_top_left[:, i]` contains the
        row and the column of the image pixel at
        the top-left of the ith crop of each
        luminance image after it is compressed
        via HEVC.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j] in this
            array is the rate associated to the compression
            of the jth luminance image via HEVC using the
            ith quantization parameter.
        numpy.ndarray
            2D array with data-type `numpy.float64`.
            The element at the position [i, j]in this
            array is the PSNR associated to the compression
            of the jth luminance image via HEVC using the
            ith quantization parameter.
    
    """
    nb_images = luminances_uint8.shape[0]
    nb_qps = qps.size
    rate = numpy.zeros((nb_qps, nb_images))
    psnr = numpy.zeros((nb_qps, nb_images))
    for i in range(nb_qps):
        qp = qps[i].item()
        path_to_storage = os.path.join(path_to_hevc_vis,
                                       'qp_{}'.format(qp))
        
        # The directory containing the reconstructed images
        # is created if it does not exist.
        if not os.path.exists(path_to_storage):
            os.makedirs(path_to_storage)
        (rate[i, :], psnr[i, :]) = compute_rate_psnr(luminances_uint8,
                                                     path_to_before_hevc,
                                                     path_to_after_hevc,
                                                     path_to_cfg,
                                                     path_to_bitstream,
                                                     qp,
                                                     path_to_storage,
                                                     list_rotation,
                                                     positions_top_left)
    return (rate, psnr)

def read_400(path, height, width, nb_frames, data_type):
    """Reads a luminance video in 4:0:0 from a binary file.
    
    Parameters
    ----------
    path : str
        Path to the binary file from which
        the luminance video in 4:0:0 is
        read. The path must end with ".yuv".
    height : int
        Height of the luminance video.
    width : int
        Width of the luminance video.
    nb_frames : int
        Number of frames in the luminance
        video. If `nb_frames` is equal to 1,
        the luminance video is viewed as a
        luminance image.
    data_type : type
        Data type of the luminance video.
        `data_type` can be equal to either
        `numpy.uint8` or `numpy.uint16`.
    
    Returns
    -------
    numpy.ndarray
        3D array with data-type `data_type`.
        Luminance video.
    
    Raises
    ------
    TypeError
        If the data type is equal to neither
        `numpy.uint8` nor `numpy.uint16`.
    RuntimeError
        If pixels are missing to read a frame.
    
    """
    if data_type not in (numpy.uint8, numpy.uint16):
        raise TypeError('The data type is equal to neither `numpy.uint8` nor `numpy.uint16`.')
    nb_pixels_per_frame = height*width
    luminance_uint8or16 = numpy.zeros((height, width, nb_frames), dtype=data_type)
    with open(path, 'rb') as file:
        for i in range(nb_frames):
        
            # In the function `numpy.fromfile`, the
            # 4th argument is optional. By default, its
            # value is "", meaning that the file is
            # treated as binary.
            vector_frame = numpy.fromfile(file,
                                          dtype=data_type,
                                          count=nb_pixels_per_frame)
            if vector_frame.size != nb_pixels_per_frame:
                raise RuntimeError('Pixels are missing to read the {}th frame.'.format(i + 1))
            luminance_uint8or16[:, :, i] = numpy.reshape(vector_frame, (height, width))
    return luminance_uint8or16

def write_400(luminance_uint8or16, path):
    """Writes a luminance video in 4:0:0 to a binary file.
    
    Parameters
    ----------
    luminance_uint8or16 : numpy.ndarray
        3D array with data-type `numpy.uint8` or `numpy.uint16`.
        Luminance video.
    path : str
        Path to binary file in which the
        luminance video in 4:0:0 is saved.
        The path must endwith ".yuv".
    
    Raises
    ------
    TypeError
        If `luminance_uint8or16.dtype` is equal to
        neither `numpy.uint8` nor `numpy.uint16`.
    ValueError
        If `luminance_uint8or16.ndim` is not equal to 3.
    OSError
        If a file already exists at `path`.
    
    """
    if luminance_uint8or16.dtype not in (numpy.uint8, numpy.uint16):
        raise TypeError('`luminance_uint8or16.dtype` is equal to neither `numpy.uint8` nor `numpy.uint16`.')
    if luminance_uint8or16.ndim != 3:
        raise ValueError('`luminance_uint8or16.ndim` is not equal to 3.')
    
    # Perhaps, another program running in
    # parallel has already created a file
    # at `path`.
    if os.path.isfile(path):
        raise OSError('"{}" already exists.'.format(path))
    with open(path, 'wb') as file:
        for i in range(luminance_uint8or16.shape[2]):
            luminance_uint8or16[:, :, i].flatten().tofile(file)


