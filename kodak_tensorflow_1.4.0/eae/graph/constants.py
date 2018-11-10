"""A library that gathers constants used in the class `EntropyAutoencoder`, the class `IsolatedDecoder` and several scripts."""

# `LR_EAE` is the learning rate of the
# parameters of the entropy autoencoder.
LR_EAE = 1.e-4

# `LR_FCT` is the learning rate of the
# parameters of the piecewise linear functions.
LR_FCT = 0.2

# `LR_BW` is the learning rate of the
# quantization bin widths.
LR_BW = 2.e-8

# In the objective function to be minimized
# over the entropy autoencoder parameters,
# `WEIGHT_DECAY_P` weights the l2-norm weight
# decay with respect to the scaled cumulated
# approximate entropy of the quantized latent
# variables and the reconstruction error.
WEIGHT_DECAY_P = 5.e-4

# `MIN_GAMMA_BETA` is involved in the projection
# of the weights and the additive coefficients of
# all GDNs/IGDNs.
MIN_GAMMA_BETA = 2.e-5

# `MIN_BW` and `MAX_BW` are involved in the
# projection of the quantization bin widths.
MIN_BW = 0.8
MAX_BW = 4.

# `NB_ITVS_PER_SIDE_INIT` is the number of
# unit intervals in the right half of the
# grid before the 1st training starts.
NB_ITVS_PER_SIDE_INIT = 10

# `NB_POINTS_PER_INTERVAL` is the number of
# sampling points per unit interval in the grid.
NB_POINTS_PER_INTERVAL = 5
LOW_PROJECTION = 1.e-6
NB_MAPS_1 = 128
NB_MAPS_2 = 128
NB_MAPS_3 = 128
WIDTH_KERNEL_1 = 9
WIDTH_KERNEL_2 = 5
WIDTH_KERNEL_3 = 5
STRIDE_1 = 4
STRIDE_2 = 2
STRIDE_3 = 2

# `STRIDE_PROD` enables to deduce the
# height of the latent variable feature maps
# directly from the height of the images we
# feed into the entropy autoencoder. It also
# enables to deduce the width of the latent
# variable feature maps directly from the
# width of these images.
STRIDE_PROD = STRIDE_1*STRIDE_2*STRIDE_3


