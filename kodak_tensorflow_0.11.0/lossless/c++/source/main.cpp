#include "tests.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "One argument is required." << std::endl;
        std::cout << "The program ends." << std::endl;
        return 1;
    }

    error_code state(success);
    if (!strcmp(argv[1], "create_divisible"))
    {
        test_create_divisible();
    }
    else if (!strcmp(argv[1], "count_nb_bits"))
    {
        test_count_nb_bits();
    }
    else if (!strcmp(argv[1], "bitstream"))
    {
        state = test_bitstream();
    }
    else if (!strcmp(argv[1], "binary_arithmetic_coder"))
    {
        state = test_binary_arithmetic_coder();
    }
    else if (!strcmp(argv[1], "read_sign"))
    {
        state = test_read_sign();
    }
    else if (!strcmp(argv[1], "read_eg0"))
    {
        state = test_read_eg0();
    }
    else if (!strcmp(argv[1], "read_truncated_unary"))
    {
        state = test_read_truncated_unary();
    }
    else if (!strcmp(argv[1], "read_signed_ueg0"))
    {
        state = test_read_signed_ueg0();
    }
    else if (!strcmp(argv[1], "compress_lossless"))
    {
        test_compress_lossless();
    }
    else
    {
        std::cout << argv[1] << " is not a function to be tested." << std::endl;
        std::cout << "The program ends." << std::endl;
        return 1;
    }

    if (state)
    {
        std::cout << "Error code: " << state << std::endl;
        return 1;
    }
    return 0;
}


