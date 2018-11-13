"""A library that contains functions for creating the Kodak test set."""

import numpy
import os
import pickle
import scipy.misc
import six.moves.urllib

import tools.tools as tls

def create_kodak(source_url, path_to_store_rgbs, path_to_kodak, path_to_list_rotation):
    """Creates the Kodak test set.
    
    The 24 Kodak RGB images are downloaded and
    converted into luminance images. Then, the
    sideways luminance images are rotated. Finally,
    the Kodak test set is filled with the luminance
    images and it is saved.
    
    Parameters
    ----------
    source_url : str
        URL of the folder that contains the
        24 Kodak RGB images.
    path_to_store_rgbs : str
        Path to the folder in which the downloaded
        24 Kodak RGB images are saved.
    path_to_kodak : str
        Path to the file in which the Kodak test
        set is saved. The path must end with ".npy".
    path_to_list_rotation : str
        Path to the file in which the list
        storing the indices of the rotated
        luminance images is saved. The path
        must end with ".pkl".
    
    Raises
    ------
    ValueError
        If a RGB image is neither
        512x768x3 nor 768x512x3.
    
    """
    if os.path.isfile(path_to_kodak) and os.path.isfile(path_to_list_rotation):
        print('"{0}" and "{1}" already exist.'.format(path_to_kodak, path_to_list_rotation))
        print('Delete them manually to recreate the Kodak test set.')
    else:
        
        # If the Kodak test set already exists, there is
        # no need to download the Kodak RGB images.
        download_option(source_url,
                        path_to_store_rgbs)
        h_kodak = 512
        w_kodak = 768
        reference_uint8 = numpy.zeros((24, h_kodak, w_kodak), dtype=numpy.uint8)
        list_rotation = []
        for i in range(24):
            path_to_file = os.path.join(path_to_store_rgbs, 'kodim' + str(i + 1).rjust(2, '0') + '.png')
            
            # The function `tls.rgb_to_ycbcr` checks that
            # the data-type of its input array is equal to
            # `numpy.uint8`. `tls.rgb_to_ycbcr` also checks
            # that its input array has 3 dimensions and its
            # 3rd dimension is equal to 3.
            luminance_uint8 = tls.rgb_to_ycbcr(scipy.misc.imread(path_to_file))[:, :, 0]
            (height_image, width_image) = luminance_uint8.shape
            if height_image == h_kodak and width_image == w_kodak:
                reference_uint8[i, :, :] = luminance_uint8
            elif width_image == h_kodak and height_image == w_kodak:
                reference_uint8[i, :, :] = numpy.rot90(luminance_uint8)
                list_rotation.append(i)
            else:
                raise ValueError('"{0}" is neither {1}x{2}x3 nor {2}x{1}x3.'.format(path_to_file, h_kodak, w_kodak))
        
        numpy.save(path_to_kodak, reference_uint8)
        with open(path_to_list_rotation, 'wb') as file:
            pickle.dump(list_rotation, file, protocol=2)

def download_option(source_url, path_to_store_rgbs):
    """Downloads the 24 Kodak RGB images.
    
    Parameters
    ----------
    source_url : str
        URL of the folder that contains the
        24 Kodak RGB images.
    path_to_store_rgbs : str
        Path to the folder in which the downloaded
        24 Kodak RGB images are saved.
    
    """
    for i in range(24):
        filename = 'kodim' + str(i + 1).rjust(2, '0') + '.png'
        path_to_file = os.path.join(path_to_store_rgbs, filename)
        if os.path.isfile(path_to_file):
            print('"{}" already exists. The image is not downloaded.'.format(path_to_file))
        else:
            six.moves.urllib.request.urlretrieve(source_url + filename,
                                                 path_to_file)
            print('Successfully downloaded "{}".'.format(filename))


