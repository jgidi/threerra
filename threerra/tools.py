#!/usr/bin/env python3

def closest_multiple(N, base : int = 16):
    """
    Return the closest multiple of 'base' to 'N'
    """
    return base * round( N / base )
