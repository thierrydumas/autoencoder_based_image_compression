# Autoencoder based image compression: can the learning be quantization independent?

This repository is a Tensorflow implementation of the paper "Autoencoder based image compression: can the learning be quantization independent?", *ICASSP 2018*.

[ICASSP 2018 paper](https://arxiv.org/abs/1802.09371) | [Project page with visualizations](https://www.irisa.fr/temics/demos/visualization_ae/visualizationAE.htm)

The code is tested on Linux and Windows.

## Prerequisites
  * Python (the code was tested using Python 2.7.9 and Python 3.6.3)
  * numpy (version >= 1.11.0)
  * tensorflow (optional GPU support), see [TensorflowInstallationWebPage](https://www.tensorflow.org/install/) (for Python 2.7.9, the code was tested using Tensorflow 0.11.0; for Python 3.6.3, the code was tested using Tensorflow 1.4.0; the code must thus work using any Tensorflow 0.x or 1.x, x being the subversion index)
  * cython
  * matplotlib
  * pillow
  * scipy
  * six
  * glymur, see [GlymurAdvancedInstallationWebPage](https://glymur.readthedocs.io/en/v0.8.14/detailed_installation.html)
  * ImageMagick, see [ImageMagickWebPage](https://www.imagemagick.org)
  
## Cloning the code
Clone this repository into the current folder.
```sh
git clone https://github.com/thierrydumas/autoencoder_based_image_compression.git
cd autoencoder_based_image_compression/kodak_tensorflow/
```

## Compilation
1. Compilation of the C++ lossless coder via Cython.
```sh
cd lossless
python setup.py build_ext --inplace
cd ../
```
2. Compilation of HEVC/H.265.
    * For Linux,
      ```sh
      cd hevc/HM-16.15/build/linux/
      make
      cd ../../../../
      ```
    * For Windows, use Visual Studio 2015 and the solution file at "hevc/HM-16.15/build/HM_vc2015.sln". For more information, see [HEVCSoftwareWebPage](https://hevc.hhi.fraunhofer.de/).

## Quick start: reproducing the main results of the paper
1. Creation of the Kodak test set containing 24 luminance images.
```sh
python creating_kodak.py
```
2. Comparison of several trained autoencoders, JPEG2000, and H.265 in terms of rate-distortion on the Kodak test set.
```sh
python reconstructing_eae_kodak.py
```

After running 2., the reconstructions of the luminance images in the Kodak test set and the rates and the PSNRs associated to the compression of the luminance images via the trained autoencoders, JPEG2000, and H.265 are stored in the folder "eae/visualization/test/checking_reconstructing/kodak/".

## Quick start: training an autoencoder
1. First of all, ImageNet images must be downloaded. In our case, it is sufficient to download the ILSVRC2012 validation images, "ILSVRC2012_img_val.tar" (6.3 GB), see [ImageNetDownloadWebPage](http://image-net.org/download). Let's say that, in your computer, the path to "ILSVRC2012_img_val.tar" is "path/to/folder_0/ILSVRC2012_img_val.tar" and you want the unpacked images to be put into the folder "path/to/folder_1/" before the script "creating_imagenet.py" preprocesses them. The creation of the ImageNet training and validaton sets of luminance images is then done via
```sh
python creating_imagenet.py path/to/folder_1/ --path_to_tar=path/to/folder_0/ILSVRC2012_img_val.tar
```
2. The training of an autoencoder on the ImageNet training set is done via the command below. 1.0 is the value of the quantization bin widths at the beginning of the training. 14000.0 is the value of the coefficient weighting the distortion term and the rate term in the objective function to be minimized over the parameters of the autoencoder. The script "training_eae_imagenet.py" enables to split the entire autoencoder training into several successive parts. The last argument 0 means that "training_eae_imagenet.py" runs the first part of the entire training. For each successive part, the last argument is incremented by 1.
```sh
python training_eae_imagenet.py 1.0 14000.0 0
```

## Full functionality
The documentation "documentation_kodak/documentation_code.html" describes all the functionalities of the code of the paper.

## A simple example
Another piece of code is a simple example for introducing the code of the paper. This piece of code is stored in the folder "svhn". Its documentation is in the file "documentation_svhn/documentation_code.html". If you feel comfortable with autoencoders, this piece of code can be skipped. Its purpose is to clarify the training of a rate-distortion optimized autoencoder. That is why a simple rate-distortion optimized autoencoder with very few hidden units is trained on tiny images (32x32 SVHN digits).

## Citing
```
@InProceedings{autoencoder_based_icassp2018,
  author = {Dumas, Thierry and Roumy, Aline and Gullemot, Christine},
  title = {Autoencoder based image compression: can the learning be quantization independent?},
  booktitle = {ICASSP},
  year = {2018}
}
