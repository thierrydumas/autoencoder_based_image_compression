#include "tests.h"

void test_create_divisible()
{
    uint32_t input(31);
    uint32_t divisor(9);
    uint32_t output(create_divisible(input, divisor));
    std::cout << output << " is the smallest 32-bit unsigned integer that is larger than " << input << " and divisible by " << divisor << "." << std::endl;
    input = 45;
    divisor = 5;
    output = create_divisible(input, divisor);
    std::cout << output << " is the smallest 32-bit unsigned integer that is larger than " << input << " and divisible by " << divisor << "." << std::endl;
    input = 101;
    divisor = 5;
    output = create_divisible(input, divisor);
    std::cout << output << " is the smallest 32-bit unsigned integer that is larger than " << input << " and divisible by " << divisor << "." << std::endl;
}

void test_count_nb_bits()
{
    uint8_t nb_bits(0);
    for (uint32_t i(0); i < 65; i++)
    {
        nb_bits = count_nb_bits(i);
        std::cout << "The binary representation of " << i << " has " << (int)nb_bits << " significant bits." << std::endl;
    }
}

error_code test_bitstream()
{
    error_code state(success);
    Bitstream bitstream(5);
    std::cout << "Bitstream size: " << bitstream.size_in_bits() << " bits." << std::endl;

    // Below is the writing step.
    std::cout << "Bitstream occupancy before the writing step: " << bitstream.occupancy_in_bits() << " bits." << std::endl;
    uint8_t byte_storing_bit(0);
    state = bitstream.write_bit(byte_storing_bit);
    if (state)
    {
        return state;
    }
    byte_storing_bit = 1;
    state = bitstream.write_bit(byte_storing_bit);
    if (state)
    {
        return state;
    }
    std::cout << "Sequence of bits written to the bitstream: 01" << std::endl;
    std::cout << "Bitstream occupancy after the writing step (before the reading step): " << bitstream.occupancy_in_bits() << " bits." << std::endl;

    // Below is the reading step.
    uint8_t storage(0);
    std::string str_bits("");
    while (!bitstream.is_entirely_read())
    {
        state = bitstream.read_bit(storage);
        if (state)
        {
            return state;
        }
        str_bits += std::to_string(storage);
    }
    std::cout << "Bitstream occupancy after the reading step: " << bitstream.occupancy_in_bits() << " bits." << std::endl;
    std::cout << "Sequence of bits read from the bitstream: " << str_bits << std::endl;
    return state;
}

error_code test_binary_arithmetic_coder()
{
    error_code state(success);
    BinaryArithmeticCoder bac(72);
    const uint8_t truncated_unary_length(20);
    const double probabilities[truncated_unary_length] = {
        0.01, 0.99, 0.9, 0.76, 0.1,
        0.01, 0.99, 0.5, 0.51, 0.2,
        0.52, 0.01, 0.1, 0.01, 0.2,
        0.90, 0.05, 0.5, 0.53, 0.2
    };

    // Below is the encoding step.
    std::cout << "Bitstream occupancy before the encoding step: " << bac.occupancy_in_bits() << " bits." << std::endl;
    uint8_t input(0);
    uint8_t idx_probability(0);
    std::string str_bits("");
    for (uint8_t i(0); i < truncated_unary_length; i++)
    {
        input = ((i >= 8) && (i <= 14)) ? 1 : 0;
        state = bac.encoding(input, probabilities[idx_probability]);
        if (state)
        {
            return state;
        }
        idx_probability++;
        str_bits += std::to_string(input);
    }
    std::cout << "Sequence of encoded bits: " << str_bits << std::endl;

    /*
    Do not forget to stop the encoding and
    start the decoding.
    */
    state = bac.stop_encoding();
    if (state)
    {
        return state;
    }
    std::cout << "Bitstream occupancy after stopping the binary arithmetic encoding (before starting the decoding step): " << bac.occupancy_in_bits() << " bits." << std::endl;
    state = bac.start_decoding();
    if (state)
    {
        return state;
    }

    // Below is the decoding step.
    uint8_t storage(0);
    idx_probability = 0;
    str_bits = "";
    for (uint8_t i(0); i < truncated_unary_length; i++)
    {
        state = bac.decoding(storage, probabilities[idx_probability]);
        if (state)
        {
            return state;
        }
        idx_probability++;
        str_bits += std::to_string(storage);
    }
    std::cout << "Bitstream occupancy after the decoding step: " << bac.occupancy_in_bits() << " bits." << std::endl;
    std::cout << "Sequence of decoded bits: " << str_bits << std::endl;
    return state;
}

error_code test_read_sign()
{
    error_code state(success);
    const uint8_t truncated_unary_length(8);
    const double probabilities[truncated_unary_length] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    LosslessCoder lossless_coder(1, truncated_unary_length, probabilities);
    std::cout << "Occupancy of the bypass bitstream before the encoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;

    // Below is the encoding step.
    const int16_t input(-21);
    state = lossless_coder.write_sign(input);
    if (state)
    {
        return state;
    }
    std::cout << "The sign of " << input << " is encoded." << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the encoding step (before the decoding step): " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;

    // Below is the decoding step.
    int16_t output(21);
    state = lossless_coder.read_sign(output);
    if (state)
    {
        return state;
    }
    std::cout << "Output: " << output << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the decoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;
    return state;
}

error_code test_read_eg0()
{
    error_code state(success);
    const uint8_t truncated_unary_length(8);
    const double probabilities[truncated_unary_length] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

    /*
    The EG0 code of 65535 contains 33 bits.
    If the size of the bypass bitstream is
    equal to 33 x 7 = 231 bits, it can store
    the EG0 seven codes.
    */
    LosslessCoder lossless_coder(231, truncated_unary_length, probabilities);
    std::cout << "Occupancy of the bypass bitstream before the encoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;
    const uint16_t test_array[7] = {0, 1, 2, 11, 128, 504, 65535};

    // Below is the encoding step.
    std::string str_integers("");
    for (uint8_t i(0); i < 7; i++)
    {
        state = lossless_coder.write_eg0(test_array[i]);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(test_array[i]) + " ";
    }
    std::cout << "Encoded unsigned integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the encoding step (before the decoding step): " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;

    // Below is the decoding step.
    uint16_t output(0);
    str_integers = "";
    for (uint8_t i(0); i < 7; i++)
    {
        state = lossless_coder.read_eg0(output);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(output) + " ";
    }
    std::cout << "Decoded unsigned integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the decoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;
    return state;
}

error_code test_read_truncated_unary()
{
    error_code state(success);
    const uint8_t truncated_unary_length(8);
    const double probabilities[truncated_unary_length] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    /*
    The truncated unary prefix of 65535 contains
    8 bits. If the size of the bitstream of the
    binary arithmetic coder is equal to 8 x 7 = 56
    bits, it can store the seven truncated unary
    prefixes. This consideration assumes that the
    binary arithmetic coder does not help compress.
    */
    LosslessCoder lossless_coder(56, truncated_unary_length, probabilities);
    std::cout << "Occupancy of the binary arithmetic coder bitstream before the encoding step: " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;
    const uint16_t test_array[7] = {0, 1, 2, 11, 128, 504, 65535};

    // Below is the encoding step.
    std::string str_integers("");
    for (uint8_t i(0); i < 7; i++)
    {
        state = lossless_coder.write_truncated_unary(test_array[i]);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(test_array[i]) + " ";
    }

    /*
    Do not forget to stop the binary arithmetic
    encoding.
    */
    state = lossless_coder.stop_bac_encoding();
    if (state)
    {
        return state;
    }
    std::cout << "Encoded unsigned integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the binary arithmetic coder bitstream after stopping the binary arithmetic encoding (before starting the decoding step): " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;

    /*
    Do not forget to start the binary arithmetic
    decoding.
    */
    state = lossless_coder.start_bac_decoding();
    if (state)
    {
        return state;
    }

    // Below is the decoding step.
    uint16_t output(0);
    str_integers = "";
    for (uint8_t i(0); i < 7; i++)
    {
        state = lossless_coder.read_truncated_unary(output);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(output) + " ";
    }
    std::cout << "Decoded unsigned integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the binary arithmetic coder bitstream after the decoding step: " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;
    return state;
}

error_code test_read_signed_ueg0()
{
    error_code state(success);
    const uint8_t truncated_unary_length(8);
    const double probabilities[truncated_unary_length] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    
    /*
    The EG0 code of 523 - 8 = 515 contains 19 bits
    and the sign of -523 costs 1 bit. If the size
    of the bypass bitstream is equal to 20 x 10 = 200
    bits, it is enough. The truncated unary prefix of
    523 contains 8 bits. If the size of the bitstream
    of the binary arithmetic coder is equal to 8 x 10 = 80
    bits, it is enough. This consideration assumes that
    the binary arithmetic coder does not help compress.
    */
    LosslessCoder lossless_coder(200, truncated_unary_length, probabilities);
    std::cout << "Occupancy of the binary arithmetic coder bitstream before the encoding step: " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;
    std::cout << "Occupancy of the bypass bitstream before the encoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;
    const int16_t test_array[10] = {0, 1, -2, -7, 8, -8, 9, -9, 127, -523};

    // Below is the encoding step.
    std::string str_integers("");
    for (uint8_t i(0); i < 10; i++)
    {
        state = lossless_coder.write_signed_ueg0(test_array[i]);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(test_array[i]) + " ";
    }

    /*
    Do not forget to stop the binary arithmetic
    encoding.
    */
    state = lossless_coder.stop_bac_encoding();
    if (state)
    {
        return state;
    }
    std::cout << "Encoded signed integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the binary arithmetic coder bitstream after the stopping the binary arithmetic encoding (before starting the decoding step): " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the encoding step (before the decoding step): " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;

    /*
    Do not forget to start the binary arithmetic
    decoding.
    */
    state = lossless_coder.start_bac_decoding();
    if (state)
    {
        return state;
    }

    // Below is the decoding step.
    int16_t output(0);
    str_integers = "";
    for (uint8_t i(0); i < 10; i++)
    {
        state = lossless_coder.read_signed_ueg0(output);
        if (state)
        {
            return state;
        }
        str_integers += std::to_string(output) + " ";
    }
    std::cout << "Decoded signed integers: " << str_integers << std::endl;
    std::cout << "Occupancy of the binary arithmetic coder bitstream after the decoding step: " << lossless_coder.occupancy_in_bits_bac() << " bits." << std::endl;
    std::cout << "Occupancy of the bypass bitstream after the decoding step: " << lossless_coder.occupancy_in_bits_bypass() << " bits." << std::endl;
    return state;
}

void test_compress_lossless()
{
    const uint32_t size(12);
    const int16_t array_input[size] = {0, -2, 0, 765, -21, 8, -439, 0, 0, 0, 0, -9};
    int16_t array_output[size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t truncated_unary_length(8);
    const double probabilities[truncated_unary_length] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    uint32_t nb_bits = compress_lossless(size,
                                         array_input,
                                         array_output,
                                         truncated_unary_length,
                                         probabilities);
    std::string str_integers_0("");
    std::string str_integers_1("");
    for (uint32_t i(0); i < size; i++)
    {
        str_integers_0 += std::to_string(array_input[i]) + " ";
        str_integers_1 += std::to_string(array_output[i]) + " ";
    }
    std::cout << "Encoded signed integers: " << str_integers_0 << std::endl;
    std::cout << "Decoded signed integers: " << str_integers_1 << std::endl;
    std::cout << "Number of bits in the global bitstream: " << nb_bits << std::endl;
}


