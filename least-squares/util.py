import numpy as np


def compute_inverse(matrix: np.array, log_output=True) -> np.array:
    """
    Calculates the inverse of a 2D matrix, A, by setting up
    the augmented matrix [A | I], where I is the identity matrix.

    (I believe this constitues Gauss-Jordan Elimination, but I am not sure).

    Assumption: that all values in the matrix are positive 
                (in some cases I guess it might still work with 0 included). 
    """
    # A: setup
    A = matrix
    nrows, ncols = A.shape
    identity_matrix = np.eye(nrows, ncols)
    augmented_matrix = np.concatenate((A, identity_matrix), axis=1)
    AI = augmented_matrix
    # B: try to get A into RREF
    diagonal_length = min(nrows, ncols)
    # for each diagonal elem --> use an index i
    for i in range(diagonal_length):
        # sort the rows at and below i, by the value in the ith col
        sub_matrix = AI[i:, :]
        sorted_sub_matrix = sub_matrix[np.argsort(sub_matrix[:, i])]
        AI = np.concatenate([AI[:i, :], sorted_sub_matrix], axis=0)
        # divide the pivot row by the pivot aka, the elem at (i, i) by itself --> replace
        AI[i, :] /= AI[i, i]
        # for the rows above/below row i, with nonzeros in the ith col:
        for row_index in range(nrows):
            if row_index != i and AI[row_index, i] != 0:
                # do the row transformation
                AI[row_index, :] += ((-1 * AI[row_index, i]) * AI[i, :])
    # if any rows are all zeros --> move them to the bottom
    for row_index in range(AI.shape[0]):
        row = AI[row_index]
        if np.array_equal(row, np.zeros(ncols,)):
            AI = np.delete(AI, row_index, axis=0)
            AI = np.append(AI, [row], axis=0)
    if log_output:
        print("Augmented matrix AI: \n", AI)
    # C: return the inverse --> we want all of (:rows, ncols+1:)
    return AI[:, ncols:]


if __name__ == "__main__":
    A = np.array(([3, 4], [-1, 6], [0, 7], [2, 5]))  # test case
    compute_inverse(A)
