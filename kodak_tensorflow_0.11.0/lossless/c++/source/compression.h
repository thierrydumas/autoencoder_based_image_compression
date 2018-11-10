/** @file */

#ifndef COMPRESSION_H
#define COMPRESSION_H

/*
Without including <string>, Visual Studio says that
`std::to_string` is not a member of `std`.
For Linux, it is useless to include <string>.
*/
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "LosslessCoder.h"

/** @brief Compresses signed integers without loss.
*
*   @details The lossless coder combines a binarizer
*            and a binary arithmetic coder.
*
*   @param size Size of `array_input` and `array_output`.
*   @param array_input Signed integers to be compressed without loss.
*   @param array_output Reconstruction of the signed integers after
*                       the compression without loss.
*   @param truncated_unary_length Length of the truncated unary prefix.
*   @param probabilities Its ith element is the probability that the
*                        ith binary decision is 0 in the truncated
*                        unary prefix.
*   @return Coding cost of the signed integers. The coding cost is
*           expressed in bits.
*   @throw invalid_argument If one of the three pointers is NULL.
*   @throw runtime_error If an error occurs during the encoding.
*   @throw runtime_error If an error occurs when stopping the
*                        binary arithmetic encoding.
*   @throw runtime_error If an error occurs when starting the
*                        binary arithmetic decoding.
*   @throw runtime_error If an error occurs during the decoding.
*/
uint32_t compress_lossless(uint32_t const& size,
                           const int16_t* const array_input,
                           int16_t* const array_output,
                           uint8_t const& truncated_unary_length,
                           const double* const probabilities);

#endif // COMPRESSION_H


