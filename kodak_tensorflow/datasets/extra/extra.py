"""A library that contains a function for creating the extra set."""

import numpy
import os
import random

import tools.tools as tls

def create_extra(source_url_inria_holidays_0, source_url_inria_holidays_1, path_to_folder_rgbs_ilsvrc2012,
                 path_to_folder_rgbs_inria_holidays, width_crop, nb_extra, path_to_extra, path_to_tar_ilsvrc2012='',
                 path_to_tar_inria_holidays_0='', path_to_tar_inria_holidays_1=''):
    """Creates the extra set.
    
    Parameters
    ----------
    source_url_inria_holidays_0 : str
        URL of the first part of the original INRIA Holidays dataset.
    source_url_inria_holidays_1 : str
        URL of the second part of the original INRIA Holidays dataset.
    path_to_folder_rgbs_ilsvrc2012 : str
        Path to the folder storing ImageNet RGB images.
    path_to_folder_rgbs_inria_holidays : str
        Path to the folder to which the original INRIA Holidays
        dataset (RGB images from the two files ".tar") is extracted.
    width_crop : int
        Width of the crop.
    nb_extra : int
        Number of luminance crops in the extra set.
    path_to_extra : str
        Path to the file in which the extra set
        is saved. The path ends with ".npy".
    path_to_tar_ilsvrc2012 : str, optional
        Path to an archive containing ImageNet RGB images.
        The default value is ''. If the path is not the
        default path, the archive is extracted to `path_to_folder_rgbs_ilsvrc2012`
        before the function starts creating the extra set.
    path_to_tar_inria_holidays_0 : str, optional
        Path to the downloaded archive containing the first part
        of the original INRIA Holidays dataset. The default value
        is ''. If the path is not the default path, the archive is
        extracted to `path_to_folder_rgbs_inria_holidays` before the
        function starts creating the extra set.
    path_to_tar_inria_holidays_1 : str, optional
        Path to the downloaded archive containing the second part
        of the original INRIA Holidays dataset. The default value
        is ''. If the path is not the default path, the archive is
        extracted to `path_to_folder_rgbs_inria_holidays` before the
        function starts creating the extra set.
    
    Raises
    ------
    RuntimeError
        If there are not enough RGB images
        to create the extra set.
    
    """
    if os.path.isfile(path_to_extra):
        print('"{}" already exists.'.format(path_to_extra))
        print('Delete it manually to recreate the extra set.')
    else:
        if path_to_tar_ilsvrc2012:
            tls.untar_archive(path_to_folder_rgbs_ilsvrc2012,
                              path_to_tar_ilsvrc2012)
        if path_to_tar_inria_holidays_0:
            is_downloaded_0 = tls.download_untar_archive(source_url_inria_holidays_0,
                                                         path_to_folder_rgbs_inria_holidays,
                                                         path_to_tar_inria_holidays_0)
            if is_downloaded_0:
                print('Successfully downloaded "{}".'.format(path_to_tar_inria_holidays_0))
            else:
                print('"{}" already exists.'.format(path_to_tar_inria_holidays_0))
                print('Delete it manually to re-download it.')
        if path_to_tar_inria_holidays_1:
            is_downloaded_1 = tls.download_untar_archive(source_url_inria_holidays_1,
                                                         path_to_folder_rgbs_inria_holidays,
                                                         path_to_tar_inria_holidays_1)
            if is_downloaded_1:
                print('Successfully downloaded "{}".'.format(path_to_tar_inria_holidays_1))
            else:
                print('"{}" already exists.'.format(path_to_tar_inria_holidays_1))
                print('Delete it manually to re-download it.')
        
        # `width_crop` has to be divisible by 16.
        luminances_uint8 = numpy.zeros((nb_extra, width_crop, width_crop, 1), dtype=numpy.uint8)
        paths_to_rgbs = group_shuffle_paths_to_rgbs(path_to_folder_rgbs_ilsvrc2012,
                                                    path_to_folder_rgbs_inria_holidays)
        i = 0
        for path_to_rgb in paths_to_rgbs:
            try:
                rgb_uint8 = tls.read_image_mode(path_to_rgb,
                                                'RGB')
                crop_uint8 = tls.crop_option_2d(tls.rgb_to_ycbcr(rgb_uint8)[:, :, 0],
                                                width_crop,
                                                False)
            except (TypeError, ValueError) as err:
                print(err)
                print('"{}" is skipped.\n'.format(path_to_rgb))
                continue
            luminances_uint8[i, :, :, 0] = crop_uint8
            i += 1
            if i == nb_extra:
                break
    
    # If the previous loop was not broken,
    # `luminances_uint8` is not full. In
    # this case, the program crashes as the
    # extra set should not contain any "zero"
    # luminance crop.
    if i != nb_extra:
        raise RuntimeError('There are not enough RGB images at "{0}" and "{1}" to create the extra set.'.format(paths_to_folders_rgbs[0], path_to_folder_inria_holidays_jpg))
    numpy.save(path_to_extra, luminances_uint8)

def group_shuffle_paths_to_rgbs(path_to_folder_rgbs_ilsvrc2012, path_to_folder_rgbs_inria_holidays):
    """Groups the paths to RGB images in a single list and shuffles the resulting list.
    
    Parameters
    ----------
    path_to_folder_rgbs_ilsvrc2012 : str
        Path to the folder storing ImageNet RGB images.
    path_to_folder_rgbs_inria_holidays : str
        Path to the folder to which the original INRIA Holidays
        dataset (RGB images from the two files ".tar") is extracted.
    
    Returns
    -------
    list
        Paths to the ImageNet RGB images and the INRIA Holidays RGB images.
    
    """
    list_names_0 = tls.clean_sort_list_strings(os.listdir(path_to_folder_rgbs_ilsvrc2012),
                                               ('jpg', 'JPEG', 'png'))[0:5000]
    paths_to_rgbs_0 = [os.path.join(path_to_folder_rgbs_ilsvrc2012, name) for name in list_names_0]
    path_to_folder_inria_holidays_jpg = os.path.join(path_to_folder_rgbs_inria_holidays,
                                                     'jpg')
    list_names_1 = tls.clean_sort_list_strings(os.listdir(path_to_folder_inria_holidays_jpg),
                                               ('jpg', 'JPEG', 'png'))
    paths_to_rgbs_1 = [os.path.join(path_to_folder_inria_holidays_jpg, name) for name in list_names_1]
    paths_to_rgbs = paths_to_rgbs_0 + paths_to_rgbs_1
    
    # To make `group_shuffle_paths_to_rgbs` deterministic, the random
    # seed should be set in the script calling `group_shuffle_paths_to_rgbs`.
    random.shuffle(paths_to_rgbs)
    return paths_to_rgbs


