#include "LosslessCoder.h"

LosslessCoder::LosslessCoder(uint32_t const& required_size_in_bits,
                             uint8_t const& truncated_unary_length,
                             const double* const probabilities) :
    m_bac(required_size_in_bits),
    m_bitstream_bypass(required_size_in_bits),
    m_truncated_unary_length(truncated_unary_length),
    m_probabilities(probabilities, probabilities + truncated_unary_length)
{}

uint32_t LosslessCoder::occupancy_in_bits_bac() const
{
    return m_bac.occupancy_in_bits();
}

uint32_t LosslessCoder::occupancy_in_bits_bypass() const
{
    return m_bitstream_bypass.occupancy_in_bits();
}

error_code LosslessCoder::write_sign(int16_t const& input)
{
    error_code state(success);
    if (input)
    {
        if (input < 0)
        {
            state = m_bitstream_bypass.write_bit(0);
        }
        else
        {
            state = m_bitstream_bypass.write_bit(1);
        }
    }
    return state;
}

error_code LosslessCoder::read_sign(int16_t& output)
{
    error_code state(success);
    if (output)
    {
        uint8_t storage(0);
        state = m_bitstream_bypass.read_bit(storage);
        if (state)
        {
            return state;
        }
        if (!storage)
        {
            output *= -1;
        }
    }
    return state;
}

error_code LosslessCoder::write_eg0(uint16_t const& input)
{
    error_code state(success);

    /*
    If `input` is equal to 2^16 - 1 = 65535, a 32-bit
    unsigned integer is needed to store `input + 1`.
    */
    const uint32_t input_plus_1(input + 1);

    /*
    The number of bits in the binary representation
    of `input_plus_1` is larger than 1 and smaller
    than 32.
    */
    const uint8_t nb_bits_minus_1(count_nb_bits(input_plus_1) - 1);
    for (uint8_t i(0); i < nb_bits_minus_1; i++)
    {
        state = m_bitstream_bypass.write_bit(1);
        if (state)
        {
            return state;
        }
    }
    state = m_bitstream_bypass.write_bit(0);
    if (state)
    {
        return state;
    }

    /*
    The precedence of ...<<... (bitwise left shift)
    is lower than the precedence of ...-...
    (subtraction).
    `suffix` is smaller than `input`.
    */
    const uint16_t suffix(input_plus_1 - (1 << nb_bits_minus_1));
    uint8_t bit_isolation(0);
    for (uint8_t i(0); i < nb_bits_minus_1; i++)
    {
        /*
        The precedence of ...&... (bitwise AND)
        is lower than the precedence of ...>>...
        (bitwise right shift).
        */
        bit_isolation = (uint8_t)(suffix >> (nb_bits_minus_1 - i - 1) & 0x1);
        state = m_bitstream_bypass.write_bit(bit_isolation);
        if (state)
        {
            return state;
        }
    }
    return state;
}

error_code LosslessCoder::read_eg0(uint16_t& output)
{
    error_code state(success);
    uint8_t storage(0);
    uint8_t nb_bits_minus_1(0);
    while (true)
    {
        state = m_bitstream_bypass.read_bit(storage);
        if (state)
        {
            return state;
        }

        if (!storage)
        {
            break;
        }
        else
        {
            nb_bits_minus_1++;
        }
    }

    /*
    `output` is set to 0 before inserting
    the read bits into it.
    */
    output = 0;
    for (uint8_t i(0); i < nb_bits_minus_1; i++)
    {
        output <<= 1;
        state = m_bitstream_bypass.read_bit(storage);
        if (state)
        {
            return state;
        }

        /*
        The precedence of ...|=... (compound assignment OR)
        is lower than the precedence of ...&...
        (bitwise AND).
        */
        output |= storage & 0x1;
    }

    /*
    The precedence of ...+=... (compound assignment
    by sum) is lower than the precedence of
    ...-... (subtraction).
    */
    output += (1 << nb_bits_minus_1) - 1;
    return state;
}

error_code LosslessCoder::write_truncated_unary(uint16_t const& input)
{
    error_code state(success);
    uint8_t i(0);
    for (i = 0; i < input; i++)
    {
        state = m_bac.encoding(1, m_probabilities.at(i));
        if (state)
        {
            return state;
        }
        if (i == m_truncated_unary_length - 1)
        {
            return state;
        }
    }

    /*
    If the program gets to the line below,
    the truncated unary prefix contains strictly
    less than `m_truncated_unary_length` 1s.
    */
    state = m_bac.encoding(0, m_probabilities.at(i));
    return state;
}

error_code LosslessCoder::read_truncated_unary(uint16_t& output)
{
    error_code state(success);

    /*
    `output` is set to 0 before
    overwriting it.
    */
    output = 0;
    uint8_t i(0);
    uint8_t storage(0);
    while (true)
    {
        state = m_bac.decoding(storage, m_probabilities.at(i));
        if (state)
        {
            return state;
        }

        if (!storage)
        {
            break;
        }
        else
        {
            output++;
        }
        if (i == m_truncated_unary_length - 1)
        {
            break;
        }
        else
        {
            i++;
        }
    }
    return state;
}

error_code LosslessCoder::write_signed_ueg0(int16_t const& input)
{
    error_code state(success);
    const uint16_t absolute_input(abs(input));
    state = write_truncated_unary(absolute_input);
    if (state)
    {
        return state;
    }
    if (absolute_input >= m_truncated_unary_length)
    {
        uint16_t difference(absolute_input - m_truncated_unary_length);
        state = write_eg0(difference);
        if (state)
        {
            return state;
        }
    }
    state = write_sign(input);
    return state;
}

error_code LosslessCoder::read_signed_ueg0(int16_t& output)
{
    error_code state(success);
    uint16_t read_absolute_value(0);
    state = read_truncated_unary(read_absolute_value);
    if (state)
    {
        return state;
    }
    if (read_absolute_value == m_truncated_unary_length)
    {
        uint16_t difference(0);
        state = read_eg0(difference);
        if (state)
        {
            return state;
        }
        read_absolute_value += difference;
    }
    output = (int16_t)read_absolute_value;
    state = read_sign(output);
    return state;
}

error_code LosslessCoder::stop_bac_encoding()
{
    return m_bac.stop_encoding();
}

error_code LosslessCoder::start_bac_decoding()
{
    return m_bac.start_decoding();
}


