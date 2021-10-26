#!/usr/bin/env python3

import numpy as np
from qiskit.result import Result
from threerra.discriminators import nearest_discriminator

def closest_multiple(N, base : int = 16):
    """
    Return the closest multiple of 'base' to 'N'
    """
    return base * round( N / base )


def get_data(result,
             discriminator=nearest_discriminator.discriminator,
             ):

    results_data = np.concatenate([result.get_memory(i).flatten()
                                   for i in range(len(result.results))])

    data = discriminator(results_data)

    return data


def get_counts(data, *args, **kwargs):
    # If data is a job result, use 'get_data' to get its data
    if isinstance(data, Result):
        data = get_data(data, *args, **kwargs)

    # Count element occurrences
    unique, counts = np.unique(data, return_counts=True)

    # Return counts as dict
    return {str(unique[i]): counts[i] for i in range(len(unique))}
