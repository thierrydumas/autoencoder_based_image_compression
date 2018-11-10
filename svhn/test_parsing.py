"""A script to test the library that contains functions for parsing command-line arguments and options."""

import argparse

import parsing.parsing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the library that contains functions for parsing command-line arguments and options.')
    parser.add_argument('int_strictly_pos',
                        help='argument to be converted into a strictly positive integer',
                        type=parsing.parsing.int_strictly_positive)
    parser.add_argument('float_strictly_pos',
                        help='argument to be converted into a strictly positive float',
                        type=parsing.parsing.float_strictly_positive)
    args = parser.parse_args()
    print('{} is a strictly positive integer.'.format(args.int_strictly_pos))
    print('{} is a strictly positive float.'.format(args.float_strictly_pos))


