"""A library that contains functions for parsing command-line arguments and options."""

import argparse

def float_strictly_positive(string):
    """Converts the string into a float.
    
    Parameters
    ----------
    string : str
        String to be converted into a float.
    
    Returns
    -------
    float
        Float resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted
        into a float.
    ArgumentTypeError
        If the float resulting from the
        conversion is not strictly positive.
    
    """
    try:
        floating_point = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into a float.'.format(string))
    if floating_point <= 0.:
        raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(floating_point))
    else:
        return floating_point

def int_positive(string):
    """Converts the string into an integer.
    
    Parameters
    ----------
    string : str
        String to be converted into an integer.
    
    Returns
    -------
    int
        Integer resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted
        into an integer.
    ArgumentTypeError
        If the integer resulting from the
        conversion is not positive.
    
    """
    try:
        integer = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into an integer.'.format(string))
    if integer < 0.:
        raise argparse.ArgumentTypeError('{} is not positive.'.format(integer))
    else:
        return integer

def int_strictly_positive(string):
    """Converts the string into an integer.
    
    Parameters
    ----------
    string : str
        String to be converted into an integer.
    
    Returns
    -------
    int
        Integer resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted
        into an integer.
    ArgumentTypeError
        If the integer resulting from the
        conversion is not strictly positive.
    
    """
    try:
        integer = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError('{} cannot be converted into an integer.'.format(string))
    if integer <= 0:
        raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(integer))
    else:
        return integer


