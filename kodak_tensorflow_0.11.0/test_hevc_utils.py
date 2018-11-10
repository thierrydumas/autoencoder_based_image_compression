"""A script to test the library that contains HEVC utilities."""

import argparse
import numpy
import os
import scipy.misc

import hevc_utils.hevc_utils as hevculs
import tools.tools as tls


class TesterHEVCUtils(object):
    """Class for testing the library that contains HEVC utilities."""
    
    def test_compress_hevc(self):
        """Tests the function `compress_hevc`.
        
        A 1st image is saved at
        "hevc_utils/pseudo_visualization/compress_hevc/luminance_before.png".
        A 2nd image is saved at
        "hevc_utils/pseudo_visualization/compress_hevc/luminance_after.png".
        The test is successful if the 2nd
        image corresponds to the 1st image
        with HEVC compression artefacts.
        
        """
        path_to_before_hevc = 'hevc_utils/temp/luminance_before_hevc.yuv'
        path_to_after_hevc = 'hevc_utils/temp/luminance_after_hevc.yuv'
        path_to_cfg = 'hevc_utils/configuration/intra.cfg'
        path_to_bitstream = 'hevc_utils/temp/bitstream.bin'
        qp = 42
        
        rgb_uint8 = scipy.misc.imread('hevc_utils/pseudo_data/rgb_nightshot.jpg')
        (height_initial, width_initial, _) = rgb_uint8.shape
        height_surplus = height_initial % 8
        width_surplus = width_initial % 8
        luminance_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[0:height_initial - height_surplus, 0:width_initial - width_surplus, 0]
        scipy.misc.imsave('hevc_utils/pseudo_visualization/compress_hevc/luminance_before.png',
                          luminance_uint8)
        luminance_before_hevc_uint8 = numpy.expand_dims(luminance_uint8, axis=2)
        luminance_after_hevc_uint8 = hevculs.compress_hevc(luminance_before_hevc_uint8,
                                                           path_to_before_hevc,
                                                           path_to_after_hevc,
                                                           path_to_cfg,
                                                           path_to_bitstream,
                                                           qp,
                                                           True)
        scipy.misc.imsave('hevc_utils/pseudo_visualization/compress_hevc/luminance_after.png',
                          numpy.squeeze(luminance_after_hevc_uint8, axis=2))
    
    def test_compute_rate_psnr(self):
        """Tests the function `compute_rate_psnr`.
        
        An image is saved at
        "hevc_utils/pseudo_visualization/compute_rate_psnr/reconstruction_0.png".
        A crop of this image is saved at
        "hevc_utils/pseudo_visualization/compute_rate_psnr/reconstruction_0_crop_0.png".
        The test is successful if the image and its
        crop are rotated compared to the image at
        "hevc_utils/pseudo_data/rgb_nightshot.jpg".
        
        """
        path_to_before_hevc = 'hevc_utils/temp/luminance_before_hevc.yuv'
        path_to_after_hevc = 'hevc_utils/temp/luminance_after_hevc.yuv'
        path_to_cfg = 'hevc_utils/configuration/intra.cfg'
        path_to_bitstream = 'hevc_utils/temp/bitstream.bin'
        qp = 42
        path_to_storage = 'hevc_utils/pseudo_visualization/compute_rate_psnr/'
        list_rotation = [0, 11, 4]
        positions_top_left = numpy.array([[300], [200]], dtype=numpy.int32)
        
        rgb_uint8 = scipy.misc.imread('hevc_utils/pseudo_data/rgb_nightshot.jpg')
        (height_initial, width_initial, _) = rgb_uint8.shape
        height_surplus = height_initial % 8
        width_surplus = width_initial % 8
        
        # The 2nd and the 3rd dimension of `luminance_uint8`
        # must be divisible by 8 as the height and the width
        # of the images inserted into HEVC must be divisible
        # by the minimum CU size.
        luminances_uint8 = numpy.expand_dims(tls.rgb_to_ycbcr(rgb_uint8)[0:height_initial - height_surplus, 0:width_initial - width_surplus, 0],
                                             axis=0)
        (rate, psnr) = hevculs.compute_rate_psnr(luminances_uint8,
                                                 path_to_before_hevc,
                                                 path_to_after_hevc,
                                                 path_to_cfg,
                                                 path_to_bitstream,
                                                 qp,
                                                 path_to_storage,
                                                 list_rotation,
                                                 positions_top_left)
        print('Rate: {}'.format(rate[0]))
        print('PSNR: {}'.format(psnr[0]))
    
    def test_evaluate_hevc(self):
        """Tests the function `evaluate_hevc`.
        
        The test is successful if, the rate for the
        1st quantization parameter is much larger
        than the rate for the 2nd quantization parameter.
        Besides, the PSNR for the 1st quantization parameter
        must be much larger than the PSNR for the 2nd
        quantization parameter.
        
        """
        path_to_before_hevc = 'hevc_utils/temp/luminance_before_hevc.yuv'
        path_to_after_hevc = 'hevc_utils/temp/luminance_after_hevc.yuv'
        path_to_cfg = 'hevc_utils/configuration/intra.cfg'
        path_to_bitstream = 'hevc_utils/temp/bitstream.bin'
        qps = numpy.array([22, 42], dtype=numpy.int32)
        path_to_hevc_vis = 'hevc_utils/pseudo_visualization/evaluate_hevc/'
        list_rotation = [0, 11, 4]
        positions_top_left = numpy.array([[300], [200]], dtype=numpy.int32)
        
        rgb_uint8 = scipy.misc.imread('hevc_utils/pseudo_data/rgb_nightshot.jpg')
        (height_initial, width_initial, _) = rgb_uint8.shape
        height_surplus = height_initial % 8
        width_surplus = width_initial % 8
        luminances_uint8 = numpy.expand_dims(tls.rgb_to_ycbcr(rgb_uint8)[0:height_initial - height_surplus, 0:width_initial - width_surplus, 0],
                                             axis=0)
        (rate, psnr) = hevculs.evaluate_hevc(luminances_uint8,
                                             path_to_before_hevc,
                                             path_to_after_hevc,
                                             path_to_cfg,
                                             path_to_bitstream,
                                             qps,
                                             path_to_hevc_vis,
                                             list_rotation,
                                             positions_top_left)
        print('1st quantization parameter: {}'.format(qps[0]))
        print('Rate for the 1st quantization parameter: {}'.format(rate[0, 0]))
        print('PSNR for the 1st quantization parameter: {}'.format(psnr[0, 0]))
        print('2nd quantization parameter: {}'.format(qps[1]))
        print('Rate for the 2nd quantization parameter: {}'.format(rate[1, 0]))
        print('PSNR for the 2nd quantization parameter: {}'.format(psnr[1, 0]))
    
    def test_read_400(self):
        """Tests the function `read_400`.
        
        An image is saved at
        "hevc_utils/pseudo_visualization/read_400.png".
        The test is successful if this image
        is identical to the image at
        "hevc_utils/pseudo_visualization/write_400.png".
        
        """
        height = 525
        width = 700
        nb_frames = 1
        data_type = numpy.uint8
        
        expanded_luminance_uint8 = hevculs.read_400('hevc_utils/pseudo_data/luminance_nightshot.yuv',
                                                    height,
                                                    width,
                                                    nb_frames,
                                                    data_type)
        luminance_uint8 = numpy.squeeze(expanded_luminance_uint8, axis=2)
        scipy.misc.imsave('hevc_utils/pseudo_visualization/read_400.png',
                          luminance_uint8)
    
    def test_write_400(self):
        """Tests the function `write_400`.
        
        An image is saved at
        "hevc_utils/pseudo_visualization/write_400.png".
        The test is successful if this image
        is identical to the image at
        "hevc_utils/pseudo_visualization/read_400.png".
        
        """
        path_to_yuv = 'hevc_utils/pseudo_data/luminance_nightshot.yuv'
        rgb_uint8 = scipy.misc.imread('hevc_utils/pseudo_data/rgb_nightshot.jpg')
        luminance_uint8 = tls.rgb_to_ycbcr(rgb_uint8)[:, :, 0]
        scipy.misc.imsave('hevc_utils/pseudo_visualization/write_400.png',
                          luminance_uint8)
        if os.path.isfile(path_to_yuv):
            print('"{}" exists. Remove it manually and restart the same test.'.format(path_to_yuv))
        else:
            hevculs.write_400(numpy.expand_dims(luminance_uint8, axis=2),
                              path_to_yuv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library that contains HEVC utilities.')
    parser.add_argument('name', help='name of the function to be tested')
    args = parser.parse_args()
    tester = TesterHEVCUtils()
    getattr(tester, 'test_' + args.name)()


