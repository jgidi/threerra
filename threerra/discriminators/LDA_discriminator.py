import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def reshape_complex_vec(vec):
    """Take in complex vector vec and return 2d array w/ real, imag entries. This is needed for the learning.
    Args:
        vec (list): complex vector of data
    Returns:
        list: vector w/ entries given by (real(vec], imag(vec))
    """
    length = len(vec)
    vec_reshaped = np.zeros((length, 2))
    for i in range(len(vec)):
        vec_reshaped[i]=[np.real(vec[i]), np.imag(vec[i])]
    return vec_reshaped


def LDA_dis(IQ_012_data, points, shots=1024, acc=False):

        # construct vector w/ 0's, 1's and 2's (for testing)
        state_012 = np.zeros(shots) # shots gives number of experiments
        state_012 = np.concatenate((state_012, np.ones(shots)))
        state_012 = np.concatenate((state_012, 2*np.ones(shots)))

        # Shuffle and split data into training and test sets
        IQ_012_train, IQ_012_test, state_012_train, state_012_test = train_test_split(IQ_012_data, state_012, test_size=0.5)

        # Set up the LDA
        LDA_012 = LinearDiscriminantAnalysis()
        LDA_012.fit(IQ_012_train, state_012_train)

        if acc==True:
            score_012 = LDA_012.score(IQ_012_test, state_012_test)
            print(score_012)

        counts = LDA_012.predict(points)

        return counts
