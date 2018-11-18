"""A library that contains a function for creating the BSDS test set."""

import numpy
import os
import pickle
import scipy.misc

import tools.tools as tls

def create_bsds(path_to_root, path_to_bsds, path_to_list_rotation, path_to_tar=''):
    """Creates the BSDS test set.
    
    100 BSDS RGB images are converted into luminance
    images. The 1st row and the 1st column of each
    luminance image are removed. Then, sideways
    luminance images are rotated. Finally, the BSDS
    test set is filled with the luminance images and
    it is saved.
    
    Parameters
    ----------
    path_to_root : str
        Path to the folder storing the original
        BSDS dataset (training RGB images and
        test RGB images).
    path_to_bsds : str
        Path to the file in which the BSDS test
        set is saved. The path must end with ".npy".
    path_to_list_rotation : str
        Path to the file in which the list
        storing the indices of the rotated
        luminance images is saved. The path
        must end with ".pkl".
    path_to_tar : str, optional
        Path to an archive containing the original
        BSDS dataset. The default value is ''. If
        the path is not the default path, the
        archive is extracted to `path_to_root`
        before the function starts creating the
        BSDS test set.
    
    Raises
    ------
    AssertionError
        If the number of BSDS RGB images to be
        read is not 100.
    ValueError
        If a RGB image is neither 481x321x3
        nor 321x481x3.
    
    """
    if os.path.isfile(path_to_bsds) and os.path.isfile(path_to_list_rotation):
        print('"{0}" and "{1}" already exist.'.format(path_to_bsds, path_to_list_rotation))
        print('Delete them manually to recreate the BSDS test set.')
    else:
        if path_to_tar:
            tls.untar_archive(path_to_root,
                              path_to_tar)
        h_bsds = 321
        w_bsds = 481
        
        # The height and the width of luminance images we
        # feed into the autoencoders must be divisible by 16.
        reference_uint8 = numpy.zeros((100, h_bsds - 1, w_bsds - 1), dtype=numpy.uint8)
        list_rotation = []
        
        # The function `os.listdir` returns a list whose order
        # depends on the OS. To make the function `create_bsds`
        # independent of the OS, the output of `os.listdir` is
        # sorted.
        list_names = clean_sort_list_strings(os.listdir(os.path.join(path_to_root, 'BSDS300/images/test/')),
                                             'jpg')
        assert len(list_names) == 100, 'The number of BSDS RGB images to be read is not 100.'
        for i in range(100):
            path_to_file = os.path.join(path_to_root, 'BSDS300/images/test/', list_names[i])
            
            # The function `tls.rgb_to_ycbcr` checks that
            # the data-type of its input array is equal to
            # `numpy.uint8`. `tls.rgb_to_ycbcr` also checks
            # that its input array has 3 dimensions and its
            # 3rd dimension is equal to 3.
            luminance_uint8 = tls.rgb_to_ycbcr(scipy.misc.imread(path_to_file))[:, :, 0]
            (height_image, width_image) = luminance_uint8.shape
            if height_image == h_bsds and width_image == w_bsds:
                reference_uint8[i, :, :] = luminance_uint8[1:h_bsds, 1:w_bsds]
            elif width_image == h_bsds and height_image == w_bsds:
                reference_uint8[i, :, :] = numpy.rot90(luminance_uint8[1:w_bsds, 1:h_bsds])
                list_rotation.append(i)
            else:
                raise ValueError('"{0}" is neither {1}x{2}x3 nor {2}x{1}x3.'.format(path_to_file, h_bsds, w_bsds))
        
        numpy.save(path_to_bsds, reference_uint8)
        with open(path_to_list_rotation, 'wb') as file:
            pickle.dump(list_rotation, file, protocol=2)

def clean_sort_list_strings(list_strings, extension):
    """Removes from the list the strings that do not end with the given extension and sorts the list.
    
    Parameters
    ----------
    list_strings : list
        List of strings.
    extension : str
        Given extension.
    
    Returns
    -------
    list
        New list which contains the strings that
        end with the given extension. This list
        is sorted.
    
    """
    list_strings_extension = [string for string in list_strings if string.endswith(extension)]
    list_strings_extension.sort()
    return list_strings_extension


