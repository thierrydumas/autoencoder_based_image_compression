#include "BinaryArithmeticCoder.h"

#define PRECISION 16

/*
The precedence of ...<<... (bitwise left shift)
is lower than the precedence of (type)...
(C-style cast).
*/
/*
The binary representation of `RANGE_MAX` is
sixteen 0s followed by sixteen 1s.
*/
#define RANGE_MAX (((uint32_t)0x1 << PRECISION) - 1)

/*
The binary representation of `RANGE_HALF` is
seventeen 0s followed by fifteen 1s.
*/
#define RANGE_HALF (RANGE_MAX >> 1)

/*
The binary representation of `RANGE_QUARTER` is
eighteen 0s followed by fourteen 1s.
*/
#define RANGE_QUARTER (RANGE_HALF >> 1)
#define RANGE_THREE_QUARTERS (3*RANGE_QUARTER)

/*
The binary representation of `MASK` is
00000000000000001000000000000000.
*/
#define MASK ((uint32_t)0x1 << (PRECISION - 1))

BinaryArithmeticCoder::BinaryArithmeticCoder(uint32_t const& required_size_in_bits) :
    m_bitstream(required_size_in_bits),
    m_low(0),
    m_middle(RANGE_HALF),
    m_high(RANGE_MAX),
    m_nb_e3(0),
    m_code(0)
{}

uint32_t BinaryArithmeticCoder::occupancy_in_bits() const
{
    return m_bitstream.occupancy_in_bits();
}

error_code BinaryArithmeticCoder::encoding(uint8_t const& input, double const& probability)
{
    error_code state(success);
    state = encode_bit(input, probability);
    if (state)
    {
        return state;
    }
    state = rescale_encoding();
    return state;
}

error_code BinaryArithmeticCoder::stop_encoding()
{
    error_code state(success);

    /*
    If the condition is met, 0 and the number of
    E3 scalings in the queue plus one 1s are
    appended to the end of the code bitstream.
    Otherwise, 1 and the number of E3 scalings in
    the queue plus one 0s are appended to the end
    of the code bitstream.
    */
    m_nb_e3++;
    if (m_low < RANGE_QUARTER)
    {
        state = m_bitstream.write_bit(0);
        if (state)
        {
            return state;
        }
        state = clear_e3_queue(0);
        if (state)
        {
            return state;
        }
    }
    else
    {
        state = m_bitstream.write_bit(1);
        if (state)
        {
            return state;
        }
        state = clear_e3_queue(1);
        if (state)
        {
            return state;
        }
    }
    reset_coder();
    return state;
}

error_code BinaryArithmeticCoder::start_decoding()
{
    error_code state(success);
    uint8_t storage(0);
    for (uint32_t i(0); i < PRECISION; i++)
    {
        if (!m_bitstream.is_entirely_read())
        {
            state = m_bitstream.read_bit(storage);
            if (state)
            {
                return state;
            }
        }
        m_code <<= 0x1;
        m_code |= storage;
    }
    return state;
}

error_code BinaryArithmeticCoder::decoding(uint8_t& storage, double const& probability)
{
    error_code state(success);
    state = decode_bit(storage, probability);
    if (state)
    {
        return state;
    }
    state = rescale_decoding();
    return state;
}

void BinaryArithmeticCoder::reset_coder()
{
    m_low = 0;
    m_middle = RANGE_HALF;
    m_high = RANGE_MAX;
    m_nb_e3 = 0;
}

error_code BinaryArithmeticCoder::update_middle(double const& probability)
{
    if (isnan(probability))
    {
        return probability_error;
    }
    if ((probability <= 0.) || (probability >= 1.))
    {
        return probability_error;
    }
    m_middle = m_low + (uint32_t)(floor(probability*(m_high - m_low)));
    return success;
}

error_code BinaryArithmeticCoder::encode_bit(uint8_t const& input, double const& probability)
{
    error_code state(success);
    state = update_middle(probability);
    if (state)
    {
        return state;
    }

    /*
    Only the last bit of the binary representation
    of `input` is encoded.
    */
    if (input & 0x1)
    {
        m_low = m_middle + 1;
    }
    else
    {
        m_high = m_middle;
    }
    return state;
}

error_code BinaryArithmeticCoder::rescale_encoding()
{
    if (m_high > RANGE_MAX || m_low > RANGE_MAX)
    {
        return precision_error;
    }
    error_code state(success);

    /*
    If the binary representation of `masked_high`
    is 00000000000000001000000000000000, `high`
    is strictly larger than `RANGE_HALF`. Otherwise,
    the binary representation of `masked_high` is
    00000000000000000000000000000000 and `high`
    is smaller than `RANGE_HALF`.
    */
    uint32_t masked_high(0);
    while (true)
    {
        masked_high = m_high & MASK;

        /*
        The precedence of ...&... (logical AND) is
        lower than the precedence of ...==...
        (relation operator =).
        */
        if (masked_high == (m_low & MASK))
        {
            if (masked_high != 0x0)
            {
                m_high -= (RANGE_HALF + 1);
                m_low -= (RANGE_HALF + 1);
            }
            m_high <<= 0x1;
            m_high |= 0x1;
            m_low <<= 0x1;

            /*
            The 17th bit in the binary representation
            of `masked_high` is shifted to the last position
            (32th position) and the result is cast from
            `uint32_t` to `uint8_t`.
            */
            uint8_t value(masked_high >> (PRECISION - 1));
            state = m_bitstream.write_bit(value);
            if (state)
            {
                return state;
            }
            state = clear_e3_queue(value);
            if (state)
            {
                return state;
            }
        }
        else if (m_low > RANGE_QUARTER && m_high <= RANGE_THREE_QUARTERS)
        {
            m_high -= (RANGE_QUARTER + 1);
            m_low -= (RANGE_QUARTER + 1);
            m_high <<= 0x1;
            m_high |= 0x1;
            m_low <<= 0x1;
            m_nb_e3 += 1;
        }
        else
        {
            break;
        }
    }
    return state;
}

error_code BinaryArithmeticCoder::decode_bit(uint8_t& storage, double const& probability)
{
    error_code state(success);
    state = update_middle(probability);
    if (state)
    {
        return state;
    }
    if (m_code >= m_low && m_code <= m_middle)
    {
        m_high = m_middle;
        storage = 0;
    }
    else if (m_code > m_middle && m_code <= m_high)
    {
        m_low = m_middle + 1;
        storage = 1;
    }
    return state;
}

error_code BinaryArithmeticCoder::rescale_decoding()
{
    error_code state(success);
    uint8_t storage(0);
    while (true)
    {
        if (m_high <= RANGE_HALF)
        {
        }
        else if (m_low > RANGE_HALF)
        {
            m_high -= (RANGE_HALF + 1);
            m_low -= (RANGE_HALF + 1);
            m_code -= (RANGE_HALF + 1);
        }
        else if (m_high <= RANGE_THREE_QUARTERS && m_low > RANGE_QUARTER)
        {
            m_high -= (RANGE_QUARTER + 1);
            m_low -= (RANGE_QUARTER + 1);
            m_code -= (RANGE_QUARTER + 1);
        }
        else
        {
            break;
        }

        if (!m_bitstream.is_entirely_read())
        {
            state = m_bitstream.read_bit(storage);
            if (state)
            {
                return state;
            }
        }

        m_high = ((m_high << 0x1) & RANGE_MAX) | 0x1;
        m_low = ((m_low << 0x1) & RANGE_MAX) | 0x0;
        m_code = ((m_code << 0x1) & RANGE_MAX) | storage;
    }
    return state;
}

error_code BinaryArithmeticCoder::clear_e3_queue(uint8_t const& value)
{
    error_code state(success);

    /*
    `inverted_bit` is 0 if the last bit of the
    binary representation of `value` is 1. It
    is 1 otherwise.
    */
    const uint8_t inverted_bit(!(value & 0x1));
    for (uint32_t i(0); i < m_nb_e3; i++)
    {
        state = m_bitstream.write_bit(inverted_bit);
        if (state)
        {
            return state;
        }
    }
    m_nb_e3 = 0;
    return state;
}


