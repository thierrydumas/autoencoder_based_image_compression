#include "compression.h"

uint32_t compress_lossless(uint32_t const& size,
                           const int16_t* const array_input,
                           int16_t* const array_output,
                           uint8_t const& truncated_unary_length,
                           const double* const probabilities)
{
    if (!array_input || !array_output || !probabilities)
    {
        throw std::invalid_argument("One of the three pointers is NULL.");
    }
    
    /*
    An upper bound for the coding cost of `array_input`
    is reached when all the elements in `array_input`
    are equal to the largest 16-bit signed integer (32767).
    In this case, for each element in `array_input`, the
    sign costs 1 bit, the truncated unary prefix costs
    `truncated_unary_length` bits and the EG0 suffix costs
    at most 31 bits. This consideration assumes that the
    binary arithmetic coder does not help compress.
    */
    uint32_t required_size_in_bits(size*std::max((uint32_t)32, (uint32_t)truncated_unary_length));
    LosslessCoder lossless_coder(required_size_in_bits,
                                 truncated_unary_length,
                                 probabilities);
    error_code state(success);
    for (uint32_t i(0); i < size; i++)
    {
        state = lossless_coder.write_signed_ueg0(array_input[i]);
        if (state)
        {
            throw std::runtime_error("Error of type " + std::to_string(state) + " during the encoding.");
        }
    }
    state = lossless_coder.stop_bac_encoding();
    if (state)
    {
        throw std::runtime_error("Error of type " + std::to_string(state) + " when stopping the binary arithmetic encoding.");
    }

    /*
    The number of bits in the bitstreams
    must be measured after stopping the binary
    arithmetic encoding and before starting the
    binary arithmetic decoding.
    */
    uint32_t nb_bits(lossless_coder.occupancy_in_bits_bac() + lossless_coder.occupancy_in_bits_bypass());

    state = lossless_coder.start_bac_decoding();
    if (state)
    {
        throw std::runtime_error("Error of type " + std::to_string(state) + " when starting the binary arithmetic decoding.");
    }
    for (uint32_t i(0); i < size; i++)
    {
        state = lossless_coder.read_signed_ueg0(array_output[i]);
        if (state)
        {
            throw std::runtime_error("Error of type " + std::to_string(state) + " during the decoding.");
        }
    }
    return nb_bits;
}


