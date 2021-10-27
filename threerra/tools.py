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
    """
    Get the data of a job result using a user-defined discriminator
    
        Args:
            result: Job's results from an experiment
            discriminator: Nearest centroid or LDA discriminator.
        
        Returns:
            Data of the job's results
    """

    results_data = np.concatenate([result.get_memory(i).flatten()
                                   for i in range(len(result.results))])

    data = discriminator(results_data)

    return data


def get_counts(data, *args, **kwargs):
    
    """
    Get the histogram data of an experiment.
    
        Args:
            result: Job's results from an experiment or data obtained from a given job.
        
        Returns:
            A dictionary that has the counts for each qubit.
    """
    
    # If data is a job result, use 'get_data' to get its data
    if isinstance(data, Result):
        data = get_data(data, *args, **kwargs)

    # Count element occurrences
    unique, counts = np.unique(data, return_counts=True)

    # Return counts as dict
    return {str(unique[i]): counts[i] for i in range(len(unique))}
