#ifndef LOSSLESS_CODER_H
#define LOSSLESS_CODER_H

#include "BinaryArithmeticCoder.h"

/** @class LosslessCoder
*   @brief Class representing the lossless coder.
*
*   @details All of its methods are public in
*            order to test each of them individually.
*/
class LosslessCoder
{
    BinaryArithmeticCoder m_bac;                  /**< Binary arithmetic coder. */
    Bitstream m_bitstream_bypass;                 /**< Bitstream for the bypass mode. */
    uint8_t m_truncated_unary_length;             /**< Length of the truncated unary prefix. */
    std::vector<double> m_probabilities;          /**< Its ith element is the probability that
                                                       the ith binary decision is 0 in the
                                                       truncated unary prefix. */

public:
    /** @brief Constructor.
    *
    *   @details Creates the binary arithmetic coder and allocates
    *            memory for the bypass bitstream.
    *
    *   @param required_size_in_bits Size of the memory to be allocated in
    *                                bits for both the bypass bitstream and
    *                                the bitstream of the binary arithmetic coder.
    *   @param truncated_unary_length Length of the truncated unary prefix.
    *   @param probabilities Its ith element is the probability that the ith
    *                        binary decision is 0 in the truncated unary prefix.
    */
    LosslessCoder(uint32_t const& required_size_in_bits,
                  uint8_t const& truncated_unary_length,
                  const double* const probabilities);

    /** @brief Returns the number of unread bits in the bitstream
    *          of the binary arithmetic coder.
    *
    *   @return Number of bits that have been written
    *           to the bitstream of the binary arithmetic
    *           coder but have not been read yet.
    */
    uint32_t occupancy_in_bits_bac() const;

    /** @brief Returns the number of unread bits in the
    *          bypass bitstream.
    *
    *   @return Number of bits that have been written
    *           to the bypass bitstream but have not
    *           been read yet.
    */
    uint32_t occupancy_in_bits_bypass() const;

    /** @brief Writes the code of the input sign to the bypass bitstream.
    *
    *   @param input
    *   @return Capacity error if the bypass bitstream does
    *           not have enough capacity to store one more bit.
    */
    error_code write_sign(int16_t const& input);

    /** @brief Reads the code of the output sign from the bypass bitstream.
    *
    *   @param output Value which is multiplied by the output sign.
    *   @return Resource error if, in the bypass bitstream, there
    *           is not bit left to be read.
    */
    error_code read_sign(int16_t& output);

    /** @brief Writes the EG0 code of the input to the bypass bitstream.
    *
    *   @param input
    *   @return Capacity error if the bypass bitstream does not
    *           have enough capacity to store one more bit.
    */
    error_code write_eg0(uint16_t const& input);

    /** @brief Reads the EG0 code of the output from the bypass bitstream.
    *
    *   @param output 16-bit unsigned integer that stores
    *                 the output.
    *   @return Resource error if, in the bypass bitstream, there
    *           is not bit left to be read.
    */
    error_code read_eg0(uint16_t& output);

    /** @brief Writes the truncated unary prefix code of the input.
    *
    *   @details The binary arithmetic coder converts the truncated
    *            unary prefix code of the input into a more compact
    *            code, which is written to its bitstream.
    *
    *   @param input
    *   @return Capacity error if the bitstream of the binary
    *           arithmetic coder does not have enough capacity
    *           to store one more bit. Probability error if
    *           a binary probability is equal to `nan` or does
    *           not belong to ]0, 1[. Precision error if, in the
    *           binary arithmetic coder, either the low value or
    *           the high value exceeds the range maximum.
    */
    error_code write_truncated_unary(uint16_t const& input);

    /** @brief Reads the truncated unary prefix code of the output.
    *
    *   @details The binary arithmetic coder reads a compact code
    *            from its bitstream, which is converted into the
    *            truncated unary prefix code of the output.
    *
    *   @param output 16-bit unsigned integer that stores
    *                 the output.
    *   @return Resource error if, in the bitstream of the binary
    *           arithmetic coder, there is no bit left to be read.
    *           Probability error if a binary probability is equal
    *           to `nan` or does not belong to ]0, 1[.
    */
    error_code read_truncated_unary(uint16_t& output);

    /** @brief Writes the UEG0 code of the input.
    *
    *   @details UEG0 is the combination of the truncated
    *            unary prefix and the EG0 suffix.
    *
    *   @param input
    *   @return Capacity error if either the bypass bitstream
    *           or the bitstream of the binary arithmetic coder
    *           does not have enough capacity to store one more
    *           bit. Probability error if a binary probability
    *           is equal to `nan` or does not belong to ]0, 1[.
    *           Precision error if, in the binary arithmetic coder,
    *           either the low value or the high value exceeds the
    *           range maximum.
    */
    error_code write_signed_ueg0(int16_t const& input);

    /** @brief Reads the UEG0 code of the output.
    *
    *   @details UEG0 is the combination of the truncated
    *            unary prefix and the EG0 suffix.
    *
    *   @param output 16-bit signed integer that stores
    *                 the output.
    *   @return Resource error if, either in the bypass bitstream
    *           or in the bitstream of the binary arithmetic coder,
    *           there is no bit left to be read. Probability error
    *           if a binary probability is equal to `nan` or does
    *           not belong to ]0, 1[.
    */
    error_code read_signed_ueg0(int16_t& output);

    /** @brief Stops irreversibly the encoding of the
    *          binary arithmetic coder.
    *
    *   @return Capacity error if the bitstream of the binary
    *           arithmetic coder does not have enough capacity
    *           to store one more bit.
    */
    error_code stop_bac_encoding();

    /** @brief Starts irreversibly the decoding of the
    *          binary arithmetic coder.
    *
    *   @return Resource error if, in the bitstream of the binary
    *           arithmetic coder, there is no bit left to be read.
    */
    error_code start_bac_decoding();
};

#endif // LOSSLESS_CODER_H


