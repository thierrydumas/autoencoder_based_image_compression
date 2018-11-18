/** @file */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cmath>

/** @enum error_code
*   @brief Enumeration of the different error codes.
*/
enum error_code : uint8_t
{
    success,           /**< Success. */
    capacity_error,    /**< Capacity error when writing to the bitstream. */
    resource_error,    /**< Resource error when reading from the bitstream. */
    precision_error,   /**< Precision error in the binary arithmetic coder. */
    probability_error  /**< Probability error in the binary arithmetic coder. */
};

/** @brief Returns the smallest 32-bit unsigned
*          integer that is larger than `input`
*          and divisible by `divisor`.
*
*   @param input
*   @param divisor Divisor of the output.
*   @return The smallest 32-bit unsigned integer
*           that is larger than `input` and divisible
*           by `divisor`.
*/
uint32_t create_divisible(uint32_t input, uint32_t divisor);

/** @brief Counts the number of significant bits in
*          the binary representation of `input`.
*
*   @param input
*   @return Number of significant bits in the binary
*           representation of `input`.
*/
uint8_t count_nb_bits(uint32_t const& input);

#endif // UTILS_H


