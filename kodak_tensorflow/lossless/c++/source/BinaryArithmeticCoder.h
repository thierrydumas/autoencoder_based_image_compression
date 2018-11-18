#ifndef BINARY_ARITHMETIC_CODER_H
#define BINARY_ARITHMETIC_CODER_H

#include "Bitstream.h"

/** @class BinaryArithmeticCoder
*   @brief Class representing the binary arithmetic coder.
*/
class BinaryArithmeticCoder
{
    Bitstream m_bitstream;   /**< Binary arithmetic coder bitstream. */
    uint32_t m_low;          /**< Low value in the full range. */
    uint32_t m_middle;       /**< Middle value in the full range. */
    uint32_t m_high;         /**< High value in the full range. */
    uint32_t m_nb_e3;        /**< Number of E3 scalings. */
    uint32_t m_code;         /**< Bit reader for decoding. */

public:
    /** @brief Constructor.
    *
    *   @details Allocates the bitstream memory and
    *            initializes the low, middle and high values
    *            in the full range.
    *
    *   @param required_size_in_bits Size of the memory to
    *                                be allocated in bits.
    */
    BinaryArithmeticCoder(uint32_t const& required_size_in_bits);

    /** @brief Returns the number of unread bits.
    *
    *   @return Number of bits that have been written
    *           to the bitstream but have not been read yet.
    */
    uint32_t occupancy_in_bits() const;

    /** @brief Encodes one bit.
    *
    *   @details The last bit of the binary representation
    *            of `input` is encoded.
    *
    *   @param input 8-bit unsigned integer that stores the
    *                bit to be encoded in the last bit of 
    *                its binary representation.
    *   @param probability Probability that the bit to be encoded is 0.
    *   @return Capacity error if the bitstream does not have enough
    *           capacity to store one more bit. Probability error if
    *           `probability` is equal to `nan` or `probability` does
    *           not belong to ]0, 1[. Precision error if either the
    *           low value or the high value exceeds the range maximum.
    */
    error_code encoding(uint8_t const& input, double const& probability);

    /** @brief Stops irreversibly the global encoding process.
    *
    *   @return Capacity error if the bitstream does not have
    *           enough capacity to store one more bit.
    */
    error_code stop_encoding();

    /** @brief Starts irreversibly the global decoding process.
    *
    *   @return Resource error if, in the bitstream, there is
    *           no bit left to be read.
    */
    error_code start_decoding();

    /** @brief Decodes one bit.
    *
    *   @param storage 8-bit unsigned integer that becomes 0
    *                  is the decoded bit is 0. It becomes 1
    *                  otherwise.
    *   @param probability Probability that the bit to be decoded is 0.
    *   @return Resource error if, in the bitstream, there is
    *           no bit left to be read. Probability error if
    *           `probability` is equal to `nan` or `probability`
    *           does not belong to ]0, 1[.
    */
    error_code decoding(uint8_t& storage, double const& probability);

private:
    
    /** @brief Resets the binary arithmetic coder.
    *
    *   @details The low, middle, high values in the full
    *            range and the number of E3 scalings are set
    *            to their value at construction time. The
    *            content of the bitstream is not modified.
    */
    void reset_coder();
    
    /** @brief Updates the middle value in the full range.
    *
    *   @param probability Probability that the bit to be
    *                      encoded is 0.
    *   @return Probability error if `probability` is equal
    *           to `nan` or `probability` does not belong to ]0, 1[.
    */
    error_code update_middle(double const& probability);
    
    error_code encode_bit(uint8_t const& input, double const& probability);
    error_code rescale_encoding();
    error_code decode_bit(uint8_t& storage, double const& probability);
    error_code rescale_decoding();
    
    /** @brief Encodes the number of E3 scalings in the queue.
    *
    *   @details After encoding the number of E3 scalings
    *            in the queue, this number is set to 0.
    *   @param 8-bit unsigned integer whose binary representation
    *          last bit is used to encode the number of E3 scalings
    *          in the queue.
    *   @return Capacity error if the bitstream does not
    *           have enough capacity to store one more bit.
    */
    error_code clear_e3_queue(uint8_t const& value);
};

#endif // BINARY_ARITHMETIC_CODER_H


