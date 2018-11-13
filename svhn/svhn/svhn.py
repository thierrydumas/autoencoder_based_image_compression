"""A library that contains functions for creating the SVHN training set, the SVHN validation set and the SVHN test set."""

import numpy
import os
import scipy.io
import six.moves.urllib

import tools.tools as tls

# The functions are sorted in
# alphabetic order.

def convert_svhn(path_to_store_mats, nb_training, nb_validation, nb_test):
    """Converts three Matlab arrays storing SVHN digits into Numpy arrays.
    
    Three Matlab arrays storing SVHN digits are loaded.
    The Matlab arrays are converted into Numpy arrays.
    Then, the SVHN digits are shuffled and divided into
    a training set, a validation set and a test set.
    
    Parameters
    ----------
    path_to_store_mats : str
        Path to the folder that stores the
        downloaded files ".mat".
    nb_training : int
        Number of SVHN digits in the training set.
    nb_validation : int
        Number of SVHN digits in the validation set.
    nb_test : int
        Number of SVHN digits in the test set.
    
    Returns
    -------
    tuple
        numpy.ndarray
            2D array with data-type `numpy.uint8`.
            Training set.
        numpy.ndarray
            2D array with data-type `numpy.uint8`.
            Validation set.
        numpy.ndarray
            2D array with data-type `numpy.uint8`.
            Test set.
    
    Raises
    ------
    AssertionError
        If `nb_training + nb_validation + nb_test`
        is too small.
    
    """
    nb_total = nb_training + nb_validation + nb_test
    pack_train = scipy.io.loadmat(os.path.join(path_to_store_mats, 'train_32x32.mat'))['X']
    pack_test = scipy.io.loadmat(os.path.join(path_to_store_mats, 'test_32x32.mat'))['X']
    nb_already_loaded = pack_train.shape[3] + pack_test.shape[3]
    assert nb_already_loaded <= nb_total, \
        '`nb_training + nb_validation + nb_test` is not larger than {}.'.format(nb_already_loaded)
    nb_extra = nb_total - nb_already_loaded
    pack_extra = scipy.io.loadmat(os.path.join(path_to_store_mats, 'extra_32x32.mat'))['X'][:, :, :, 0:nb_extra]
    clutter = numpy.random.permutation(nb_total)
    
    # The SVHN digits in `pack_train`, those in `pack_test`
    # and those in `pack_extra` are mixed together.
    packs = numpy.concatenate((pack_train, pack_test, pack_extra), axis=3)[:, :, :, clutter]
    
    # There is a split to create a training
    # set, a validation set and a test set.
    training_uint8 = tls.images_to_rows(packs[:, :, :, 0:nb_training])
    validation_uint8 = tls.images_to_rows(packs[:, :, :, nb_training:nb_training + nb_validation])
    test_uint8 = tls.images_to_rows(packs[:, :, :, nb_training + nb_validation:nb_total])
    return (training_uint8, validation_uint8, test_uint8)

def create_svhn(source_url, path_to_store_mats, nb_training, nb_validation, nb_test, paths_to_outputs):
    """Creates the SVHN training set, the SVHN validation set and the SVHN test set.
    
    Three Matlab arrays storing SVHN digits are loaded.
    The Matlab arrays are converted into Numpy arrays.
    Then, the SVHN digits are shuffled and divided into
    a training set, a validation set and a test set.
    The three sets are saved. Finally, two preprocessing
    tools are computed and saved.
    
    Parameters
    ----------
    source_url : str
        URL of the SVHN website.
    path_to_store_mats : str
        Path to the folder that stores the
        downloaded files ".mat".
    nb_training : int
        Number of SVHN digits in the training set.
    nb_validation : int
        Number of SVHN digits in the validation set.
    nb_test : int
        Number of SVHN digits in the test set.
    paths_to_outputs : tuple
        The 1st string in this tuple is the path to
        the saved training set. The 2nd string is the
        path to the saved validation set. The 3rd
        string is the path to the saved test set.
        The 4th and the 5th string are the paths to
        respectively the saved 1st preprocessing
        tool and the saved 2nd preprocessing tool.
        Each path must end with ".npy".
    
    Raises
    ------
    AssertionError
        If `len(paths_to_outputs)` is not equal to 5.
    
    """
    assert len(paths_to_outputs) == 5, \
        '`len(paths_to_outputs)` is not equal to 5.'
    if all([os.path.isfile(path_to_output) for path_to_output in paths_to_outputs]):
        print('The SVHN training set, the SVHN validation set, the SVHN test set and the two preprocessing tools already exist.')
        print('Delete them manually to recreate them.')
    else:
        
        # If the the training set, the validation set
        # and the test set exist, there is no need to
        # downloaded the SVHN RGB digits.
        names_sets = (
            'train_32x32.mat',
            'test_32x32.mat',
            'extra_32x32.mat'
        )
        for name_set in names_sets:
            download_option(source_url,
                            path_to_store_mats,
                            name_set)
        (training_uint8, validation_uint8, test_uint8) = convert_svhn(path_to_store_mats,
                                                                      nb_training,
                                                                      nb_validation,
                                                                      nb_test)
        numpy.save(paths_to_outputs[0],
                   training_uint8)
        numpy.save(paths_to_outputs[1],
                   validation_uint8)
        numpy.save(paths_to_outputs[2],
                   test_uint8)
        (mean_training, std_training) = std_mean_chunks(training_uint8, 20)
        numpy.save(paths_to_outputs[3],
                   mean_training)
        numpy.save(paths_to_outputs[4],
                   std_training)

def download_option(source_url, path_to_store_mats, filename):
    """Downloads the file ".mat" from the source URL if it does not already exist.
    
    Parameters
    ----------
    source_url : str
        URL of the SVHN website.
    path_to_store_mats : str
        Path to the folder that stores the
        downloaded files ".mat".
    filename : str
        Name of the file ".mat" to be downloaded.
    
    """
    path_to_file = os.path.join(path_to_store_mats, filename)
    if os.path.isfile(path_to_file):
        print('"{}" already exists. It is not downloaded.'.format(filename))
    else:
        six.moves.urllib.request.urlretrieve(source_url + filename,
                                             path_to_file)
        print('Successfully downloaded "{}".'.format(filename))

def preprocess_svhn(images_uint8, mean_training, std_training):
    """Centers each image and reduces it.
    
    Parameters
    ----------
    images_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Images before the preprocessing.
        `images_uint8[i, :]` contains the ith image.
    mean_training : numpy.ndarray
        1D array with data-type `numpy.float64`.
        Mean of each pixel over all training images.
    std_training : numpy.float64
        Mean of the standard deviation of each
        pixel over all training images.
    
    Returns
    -------
    numpy.ndarray
        2D array with data-type `numpy.float64`.
        Images after the preprocessing.
    
    Raises
    ------
    AssertionError
        If `images_uint8.dtype` is not equal to `numpy.uint8`.
    AssertionError
        If `images_uint8.ndim` is not equal to 2.
    
    """
    assert images_uint8.dtype == numpy.uint8, \
        '`images_uint8.dtype` is not equal to `numpy.uint8`.'
    assert images_uint8.ndim == 2, \
        '`images_uint8.ndim` is not equal to 2.'
    
    # In the subtraction below, the data-type of the
    # left term is `numpy.uint8` and that of the right
    # term is `numpy.float64`. The subtraction actually
    # involves a copy of the left term with data-type
    # `numpy.float64` and the right term.
    return (images_uint8 - numpy.tile(mean_training, (images_uint8.shape[0], 1)))/std_training

def std_mean_chunks(images_uint8, nb_chunks):
    """Computes two preprocessing tools from the set of images.
    
    The set of images is split into chunks
    to compute the two preprocessing tools.
    
    Parameters
    ----------
    images_uint8 : numpy.ndarray
        2D array with data-type `numpy.uint8`.
        Set of images from which the two preprocessing
        tools are computed. `images_uint8[i, :]`
        contains the ith image.
    nb_chunks : int
        Number of chunks.
            
    Returns
    -------
    tuple
        numpy.ndarray
            1D array with data-type `numpy.float64`.
            Mean of each pixel over all images.
        numpy.float64
            Mean of the standard deviation of each
            pixel over all images.
    
    Raises
    ------
    AssertionError
        If `images_uint8.dtype` is not equal to `numpy.uint8`.
    AssertionError
        If `images_uint8.ndim` is not equal to 2.
    AssertionError
        If `images_uint8.shape[0]` is not divisible by `nb_chunks`.
    
    """
    assert images_uint8.dtype == numpy.uint8, \
        '`images_uint8.dtype` is not equal to `numpy.uint8`.'
    assert images_uint8.ndim == 2, \
        '`images_uint8.ndim` is not equal to 2.'
    (nb_images, nb_visible) = images_uint8.shape
    assert nb_images % nb_chunks == 0, \
        '`images_uint8.shape[0]` is not divisible by `nb_chunks`.'
    chunk_size = nb_images//nb_chunks
    accumulation_0 = numpy.zeros(nb_visible)
    for i in range(nb_chunks):
        accumulation_0 += numpy.mean(images_uint8[i*chunk_size:(i + 1)*chunk_size, :], axis=0)
    mean_training = accumulation_0/nb_chunks
    accumulation_1 = numpy.zeros(nb_visible)
    
    # In the subtraction below, the data-type of the
    # left term is `numpy.uint8` and that of the right
    # term is `numpy.float64`. The subtraction actually
    # involves a copy of the left term with data-type
    # `numpy.float64` and the right term.
    for i in range(nb_chunks):
        accumulation_1 += numpy.mean((images_uint8[i*chunk_size:(i + 1)*chunk_size, :] -
            numpy.tile(mean_training, (chunk_size, 1)))**2, axis=0)
    std_training = numpy.mean(numpy.sqrt(accumulation_1/nb_chunks))
    return (mean_training, std_training)


