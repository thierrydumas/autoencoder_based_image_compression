#include "utils.h"

uint32_t create_divisible(uint32_t input, uint32_t divisor)
{
    const uint32_t remainder(input % divisor);
    if (remainder)
    {
        input += divisor - remainder;
    }
    return input;
}

uint8_t count_nb_bits(uint32_t const& input)
{
    if (input)
    {
        /*
        The precedence of ...+... (addition)
        is lower than the precedence of
        (type)... (C-style cast).
        */
        return (uint8_t)(floor(log2((double)input)) + 1);
    }
    else
    {
        return 1;
    }
}


