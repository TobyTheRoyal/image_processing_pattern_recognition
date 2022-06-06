import numpy as np
import imageio
from scipy.ndimage.filters import gaussian_filter
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from numpy.linalg import eigh
import math_tools


def diffusion_tensor(im, sigma_g, sigma_u, alpha, gamma, nabla):
    """
    Implement the diffusion tensor computation (Eq. 9)
    """

    M, N = im.shape
    im_filtered = gaussian_filter(im, sigma_u).flatten()

    u_tilde = nabla @ im_filtered
    u_x_tilde = u_tilde[:im_filtered.size].reshape((M, N))
    u_y_tilde = u_tilde[im_filtered.size:].reshape((M, N))

    S = np.zeros((im_filtered.size, 2, 2))
    S[:, 0, 0] = gaussian_filter(u_x_tilde ** 2, sigma_g).flatten()
    S[:, 1, 0] = gaussian_filter(u_x_tilde * u_y_tilde, sigma_g).flatten()
    S[:, 1, 1] = gaussian_filter(u_y_tilde ** 2, sigma_g).flatten()

    mu, v = eigh(S)
    v = np.flip(v, 2)
    lam = np.zeros_like(S)
    lam[:, 0, 0] = alpha
    lam[:, 1, 1] = alpha + (1 - alpha) * (1 - np.exp(-(mu[:, 0] - mu[:, 1]) ** 2 / (2 * gamma ** 2)))
    D = v @ lam @ v.transpose((0, 2, 1))

    return sp.bmat([[sp.diags(D[:, 0, 0]), sp.diags(D[:, 1, 0])],
                 [sp.diags(D[:, 1, 0]), sp.diags(D[:, 1, 1])]], format='csr')


def coherence_enhancing_diffusion(
    image, sigma_g, sigma_u,
    alpha, gamma, tau, end_time
):
    time = 0
    U_t = image.flatten()
    nabla = math_tools.spnabla_hp(image.shape[0], image.shape[1])
    while time < end_time:
        time += tau
        print(f'{time=}')
        D = diffusion_tensor(
            U_t.reshape(image.shape), sigma_g, sigma_u, alpha, gamma, nabla
        )
        mat = sp.eye(U_t.shape[0], format='csr') + tau * nabla.T @ D @ nabla
        U_t = spsolve(mat, U_t)
    return np.reshape(U_t, image.shape)


if __name__ == "__main__":
    sigma_g = 1.5
    sigma_u = 0.7
    alpha = 0.0005
    gamma = 0.0001
    tau = 5
    endtime = 100
    input = imageio.imread("input.jpg").mean(axis=2)
    output = coherence_enhancing_diffusion(
        input, sigma_g, sigma_u, alpha, gamma, tau, endtime
    )
    imageio.imsave('./output.jpg', output)
