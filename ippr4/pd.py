import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from math_tools import spnabla


def FT(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))


def IFT(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm='ortho'))


def prox_g(u, tau, mask):
    '''
    Implement Eq. (18)
    '''
    return IFT((FT(u.reshape(O, M, N)) + tau * f) / (np.ones_like(mask) + tau * mask)).flatten()


def prox_fstar(p, sigma):
    '''
    Implement Eq. (20)
    '''
    scaled_norm = np.linalg.norm(np.abs(p).reshape(2 * O, M, N), axis=0) / (sigma * lamda)
    scaled_norm[scaled_norm < 1] = 1
    return (p.reshape(2 * O, M, N) / scaled_norm).flatten()


def energy(u, lamda, nabla):
    r = np.sum(np.sqrt(np.sum(np.abs(nabla @ u).reshape(O * 2, M, N) ** 2, 0)))
    d = np.sum(np.abs(FT(u.reshape(O, M, N)) * mask - f) ** 2) / 2
    return lamda * r + d


def reconstruction(f, mask, lamda, sigma, tau, max_iter, nabla):
    u = np.zeros(O * M * N)
    p = np.zeros(O * 2 * M * N)
    # fig, ax = plt.subplots()
    # plt.ion()

    for i in range(max_iter):
        u_old = u.copy()
        '''
        Implement the primal-dual iteration (12)
        '''
        u = prox_g(u - tau * nabla.T @ p, tau, mask)
        p = prox_fstar(p + sigma * nabla @ (2 * u - u_old), sigma)
        print(i, energy(u, lamda, nabla))
        # if i % 10 == 0:
        #     ax.clear()
        #     ax.imshow(np.sqrt(np.sum(np.abs(u).reshape(O, M, N) ** 2, 0)))
        #     fig.canvas.draw()
        #     plt.pause(0.1)

    # plt.ioff()
    # plt.show()
    plt.figure(figsize=(6, 10))
    plt.imshow(np.sqrt(np.sum(np.abs(u).reshape(O, M, N) ** 2, 0)))
    plt.title(f'Reconstruction after {max_iter} iterations')
    plt.savefig(f'recon_{lamda:.0e}.png', dpi=300)


if __name__ == '__main__':
    normalization = 0.0005315362762484158
    raw_data = np.load('./kspace.npy') / normalization
    O, M, N = raw_data.shape
    mask = np.fft.fftshift(np.load('./mask.npy'))

    # fully sampled reference
    reference = np.sum(np.abs(IFT(raw_data)), 0)
    plt.figure(figsize=(6, 10))
    plt.title('Reference image')
    plt.imshow(reference)
    plt.savefig('reference.png', dpi=300)

    f = raw_data * mask
    undersampled = np.sum(np.abs(IFT(f)), 0)
    plt.figure(figsize=(6, 10))
    plt.title('Undersampled input image')
    plt.imshow(undersampled)
    plt.savefig('input.png', dpi=300)

    max_iter = 100
    lamda = 5e-2
    nabla_xy = spnabla(M, N)
    nabla = sp.block_diag([nabla_xy] * O)

    # Lipschitz constant of nabla operator and typical choices for tau, sigma
    L = 8
    tau = 1 / np.sqrt(L)
    sigma = 1 / np.sqrt(L)


    lamdas = [5e-6, 5e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1, 5, 50]
    for lamda in lamdas:
        print(f'{lamda:.0e}')
        reconstruction(f, mask, lamda, sigma, tau, max_iter, nabla)