/** @file */

#ifndef TESTS_H
#define TESTS_H

#include "compression.h"

/** @brief Tests the function `create_divisible` in the file "utils.h". */
void test_create_divisible();

/** @brief Tests the function `count_nb_bits` in the file "utils.h". */
void test_count_nb_bits();

/** @brief Tests all the methods of class `Bitstream` in the file "Bitstream.h". */
error_code test_bitstream();

/** @brief Tests all the methods of class `BinaryArithmeticCoder` in the file "BinaryArithmeticCoder.h". */
error_code test_binary_arithmetic_coder();

/** @brief Tests the method `LosslessCoder::read_sign` in the file "LosslessCoder.h". */
error_code test_read_sign();

/** @brief Tests the method `LosslessCoder::read_eg0` in the file "LosslessCoder.h". */
error_code test_read_eg0();

/** @brief Tests the method `LosslessCoder::read_truncated_unary` in the file "LosslessCoder.h". */
error_code test_read_truncated_unary();

/** @brief Tests the method `LosslessCoder::read_signed_ueg0` in the file "LosslessCoder.h". */
error_code test_read_signed_ueg0();

/** @brief Tests the function `compress_lossless` in the file "compression.h". */
void test_compress_lossless();

#endif // TESTS_H


