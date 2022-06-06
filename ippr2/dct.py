import torch as th
import math


def mat1d(n):
    '''
    Implement the 1D DCT matrix
    '''
    temp = th.arange(n, dtype=th.float64)
    unscaled = th.cos(th.ger(temp, math.pi/n * (temp+0.5)))  # th.ger = outer product
    unscaled[0, :] = unscaled[0, :] / math.sqrt(2)
    return math.sqrt(2/n) * unscaled


def mat2d(n):
    '''
    Implement the 2D DCT matrix as the kronecker product of two 1D DCT matrices
    '''
    temp = mat1d(n)
    return th.ger(temp.view(-1), temp.view(-1)).reshape(temp.size() + temp.size()).permute([2, 0, 3, 1])\
        .reshape(temp.size()[0]**2, temp.size()[1]**2)


if __name__ == '__main__':
    th.set_printoptions(4, sci_mode=False)
    n = 3
    C = mat2d(n)
    # C.T @ C should be the (n^2 \times n^2) identity matrix -> this should be 0
    print(th.sum((C.T @ C) - th.eye(3**2)))
