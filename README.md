# Autoencoder based image compression: can the learning be quantization independent?

This repository is a Tensorflow implementation of the paper **"Autoencoder based image compression: can the learning be quantization independent?"**, **ICASSP 2018**.

[ICASSP 2018 paper](https://arxiv.org/abs/1802.09371) | [Project page with visualizations](https://www.irisa.fr/temics/demos/visualization_ae/visualizationAE.htm)

If you use Tensorflow 0.x, x being the subversion index, select the code in the folder "kodak_tensorflow_0.11.0". If you use Tensorflow 1.x, select the code in the folder "kodak_tensorflow_1.4.0".

## Dependencies
  * Python (the code was tested using Python 2.7.9 and Python 3.6.3)
  * numpy (version >= 1.11.0)
  * Tensorflow with GPU support [TensorflowWebPage](https://www.tensorflow.org/install/) (for Python 2.7.9, the code was tested using Tensorflow 0.11.0; for Python 3.6.3, the code was tested using Tensorflow 1.4.0)
  * cython
  * matplotlib
  * scipy
  * six
  * glymur (OpenJPEG is required, see [GlymurAdvancedInstallationWebPage](https://glymur.readthedocs.io/en/v0.8.7/detailed_installation.html))
  * ImageMagick (see [ImageMagickWebPage](https://www.imagemagick.org))
