"""A library that contains a function for creating the extra set."""

import numpy
import os

import tools.tools as tls

def create_extra(path_to_root, width_crop, nb_extra, path_to_extra):
    """Creates the extra set.
    
    RGB images are converted into luminance images.
    Then, the luminance images are cropped. Finally,
    the extra set is filled with the luminance crops
    and it is saved.
    
    Parameters
    ----------
    path_to_root : str
        Path to the folder containing RGB images.
    width_crop : int
        Width of the crop.
    nb_extra : int
        Number of luminance crops in the
        extra set.
    path_to_extra : str
        Path to the file in which the extra set
        is saved. The path must end with ".npy".
    
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
        luminances_uint8 = numpy.zeros((nb_extra, width_crop, width_crop, 1), dtype=numpy.uint8)
        list_names = os.listdir(path_to_root)
        i = 0
        extensions = ('jpg', 'JPEG', 'png')
        for name in list_names:
            if name.endswith(extensions):
                path_to_rgb = os.path.join(path_to_root, name)
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
            raise RuntimeError('There are not enough RGB images at "{}" to create the extra set.'.format(path_to_root))
        numpy.save(path_to_extra, luminances_uint8)


