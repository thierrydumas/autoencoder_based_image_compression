#include "Bitstream.h"

Bitstream::Bitstream(uint32_t const& required_size_in_bits) :
    m_read_index(0),
    m_write_index(0),
    m_data(create_divisible(required_size_in_bits, 8) >> 3)
{}

uint32_t Bitstream::size_in_bits() const
{
    /*
    Three bit shifts to the left is
    equivalent to the multiplication by 8.
    This is not true if the first 1 from the
    left falls off the left edge.
    */
    return m_data.size() << 3;
}

uint32_t Bitstream::occupancy_in_bits() const
{
    return m_write_index - m_read_index;
}

bool Bitstream::is_entirely_read() const
{
    return m_write_index == m_read_index;
}

error_code Bitstream::write_bit(uint8_t const& byte_storing_bit)
{
    if (m_write_index + 1 > size_in_bits())
    {
        return capacity_error;
    }
    const uint32_t destination_byte(m_write_index >> 3);
    const uint8_t destination_bit(m_write_index % 8);
    uint8_t& current_byte(m_data.at(destination_byte));

    /*
    The unused buffer memory can contain anything.
    The target bit is set to 0 before the writing
    starts.
    */
    /*
    The precedence of &= (coumpound assignment by
    bitwise AND) is lower than the precedence of
    ~... (bitwise NOT).
    */
    current_byte &= ~(0x1 << destination_bit);

    /*
    The last bit of the binary representation of
    `byte_storing_bit` overwrites the target bit.
    */
    current_byte |= ((byte_storing_bit & 0x1) << destination_bit);
    m_write_index++;
    return success;
}

error_code Bitstream::read_bit(uint8_t& storage)
{
    if (m_read_index >= m_write_index)
    {
        return resource_error;
    }
    const uint32_t source_byte(m_read_index >> 3);
    const uint8_t source_bit(m_read_index % 8);
    
    /*
    The last bit of the binary representation
    of `storage` is first set to 0. Then
    it takes the value of the read bit.
    */
    storage &= 0xFE;
    storage |= ((m_data.at(source_byte) >> source_bit) & 0x1);
    m_read_index++;
    return success;
}


