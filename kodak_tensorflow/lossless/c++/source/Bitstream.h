#ifndef BITSTREAM_H
#define BITSTREAM_H

#include <vector>

#include "utils.h"

/** @class Bitstream
*   @brief Class representing the bitstream.
*/
class Bitstream
{
    uint32_t m_read_index;          /**< Index of the last read bit. */
    uint32_t m_write_index;         /**< Index of the last written bit. */
    std::vector<uint8_t> m_data;    /**< Buffer storing the bits. */

public:
    /** @brief Constructor.
    *
    *   @details Allocates the bitstream memory.
    *
    *   @param required_size_in_bits Size of the memory to be
    *                                allocated in bits.
    */
    Bitstream(uint32_t const& required_size_in_bits);

    /** @brief Returns the size of the memory.
    *
    *   @return Size of the memory in bits.
    */
    uint32_t size_in_bits() const;

    /** @brief Returns the number of unread bits.
    *
    *   @return Number of bits that have been written but
    *           have not been read yet.
    */
    uint32_t occupancy_in_bits() const;

    /** @brief Have all the bits already been read?
    *
    *   @return True if all the bits have already been read.
    */
    bool is_entirely_read() const;

    /** @brief Writes one bit to the bitstream.
    *
    *   @param byte_storing_bit 8-bit unsigned integer that stores
    *                           the bit to be written to the bitstream
    *                           in the last bit of its binary representation.
    *   @return Capacity error if the bitstream does not
    *           have enough capacity to store one more bit.
    */
    error_code write_bit(uint8_t const& byte_storing_bit);

    /** @brief Reads one bit from the bitstream.
    *
    *   @param storage 8-bit unsigned integer that stores the read bit
    *                  in the last bit of its binary representation.
    *   @return Resource error if, in the bitstream, there is no bit
    *           left to be read.
    */
    error_code read_bit(uint8_t& storage);
};

#endif // BITSTREAM_H


