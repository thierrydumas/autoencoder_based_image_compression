# Autoencoder based image compression: can the learning be quantization independent?

This repository is a Tensorflow implementation of the paper **"Autoencoder based image compression: can the learning be quantization independent?"**, **ICASSP 2018**.

[ICASSP 2018 paper](https://arxiv.org/abs/1802.09371) | [Project page with visualizations](https://www.irisa.fr/temics/demos/visualization_ae/visualizationAE.htm)

## Prerequisites
  * Python (the code was tested using Python 2.7.9 and Python 3.6.3)
  * numpy (version >= 1.11.0)
  * Tensorflow with GPU support, see [TensorflowWebPage](https://www.tensorflow.org/install/) (for Python 2.7.9, the code was tested using Tensorflow 0.11.0; for Python 3.6.3, the code was tested using Tensorflow 1.4.0)
  * cython
  * matplotlib
  * scipy
  * six
  * glymur, see [GlymurAdvancedInstallationWebPage](https://glymur.readthedocs.io/en/v0.8.7/detailed_installation.html)
  * ImageMagick, see [ImageMagickWebPage](https://www.imagemagick.org)
  
## Cloning the code
Clone this repository into the current folder.
```sh
git clone https://github.com/thierrydumas/autoencoder_based_image_compression.git
```
If your version of Tensorflow is 0.x, x being the subversion index, use the code in the folder "kodak_tensorflow_0.11.0".
```sh
cd kodak_tensorflow_0.11.0
```
If your version of Tensorflow is 1.x, use the code in the folder "kodak_tensorflow_1.4.0".
```sh
cd kodak_tensorflow_1.4.0
```

## Compilation
0. Compilation of the C++ lossless coder via Cython.
```sh
cd lossless
python setup.py build_ext --inplace
cd ../
```
1. Compilation of HEVC/H.265:
    * For Linux, use the Makefile at "HM-16.15/build/linux/makefile".
    * For Windows, use Visual Studio 2015 and the solution file at "HM-16.15/build/HM_vc2015.sln". For more information, see [HEVCSoftwareWebPage](https://hevc.hhi.fraunhofer.de/).

