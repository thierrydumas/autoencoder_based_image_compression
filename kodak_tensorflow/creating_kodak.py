"""A script to create the Kodak test set."""

import argparse
import numpy

import kodak.kodak
import tools.tools as tls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the Kodak test set.')
    parser.parse_args()
    
    source_url = 'http://r0k.us/graphics/kodak/kodak/'
    path_to_store_rgbs = 'kodak/data'
    path_to_kodak = 'kodak/results/kodak.npy'
    path_to_list_rotation = 'kodak/results/list_rotation.pkl'
    
    kodak.kodak.create_kodak(source_url,
                             path_to_store_rgbs,
                             path_to_kodak,
                             path_to_list_rotation)
    reference_uint8 = numpy.load(path_to_kodak)
    tls.save_image('kodak/visualization/luminance_3.png',
                   reference_uint8[2, :, :])
    tls.save_image('kodak/visualization/luminance_11.png',
                   reference_uint8[10, :, :])


