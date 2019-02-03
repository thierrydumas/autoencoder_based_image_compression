"""A script to test the libraries in the folder "datasets"."""

import argparse
import numpy
import os

import datasets.extra.extra
import datasets.imagenet.imagenet
import tools.tools as tls


class TesterDatasets(object):
    """Class for testing the libraries in the folder "datasets"."""
    
    def test_create_extra(self):
        """Tests the function `create_extra` in the file "datasets/extra/extra.py".
        
        An image is saved at
        "lossless/pseudo_visualization/create_extra/crop_i.png",
        i in {0, 1, 2}.
        The test is successful if each saved
        image is the luminance central crop of
        a different RGB image in the folder
        "datasets/extra/pseudo_data/".
        
        """
        width_crop = 256
        nb_extra = 3
        path_to_extra = 'datasets/extra/pseudo_data/pseudo_extra.npy'
        
        datasets.extra.extra.create_extra('datasets/extra/pseudo_data/',
                                          width_crop,
                                          nb_extra,
                                          path_to_extra)
        pseudo_extra = numpy.load(path_to_extra)
        for i in range(nb_extra):
            tls.save_image('datasets/extra/pseudo_visualization/crop_{}.png'.format(i),
                           pseudo_extra[i, :, :, 0])
    
    def test_create_imagenet(self):
        """Tests the function `create_imagenet` in the file "datasets/imagenet/imagenet.py".
        
        An image is saved at
        "datasets/imagenet/pseudo_visualization/create_imagenet/training_i.png",
        i in {0, 1}. An image is saved at
        "datasets/imagenet/pseudo_visualization/create_imagenet/validation_i.png",
        i in {0, 1}.
        The test is successful if each saved image is
        the luminance crop of a different RGB image in
        the folder "datasets/imagenet/pseudo_data/". Besides,
        if the saved image is named "training_...", it must
        be a random crop. If the saved image is named
        "validation_...", it must be a central crop.
        
        """
        width_crop = 192
        nb_training = 2
        nb_validation = 2
        path_to_training = 'datasets/imagenet/pseudo_data/pseudo_training_data.npy'
        path_to_validation = 'datasets/imagenet/pseudo_data/pseudo_validation_data.npy'
        
        # The images in the folder "datasets/imagenet/pseudo_data/"
        # are large. Therefore, none of them is dumped.
        datasets.imagenet.imagenet.create_imagenet('datasets/imagenet/pseudo_data/',
                                                   width_crop,
                                                   nb_training,
                                                   nb_validation,
                                                   path_to_training,
                                                   path_to_validation)
        pseudo_training_data = numpy.load(path_to_training)
        path_to_folder_vis = 'datasets/imagenet/pseudo_visualization/'
        for i in range(nb_training):
            tls.save_image(os.path.join(path_to_folder_vis, 'training_{}.png'.format(i)),
                           pseudo_training_data[i, :, :, 0])
        pseudo_validation_data = numpy.load(path_to_validation)
        for i in range(nb_validation):
            tls.save_image(os.path.join(path_to_folder_vis, 'validation_{}.png'.format(i)),
                           pseudo_validation_data[i, :, :, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the libraries in the folder "datasets".')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    
    tester = TesterDatasets()
    getattr(tester, 'test_' + args.name)()


