# Do not import any other modules
import imageio
import numpy as np
# You may use **one** filter of the following module to implement the
# ```compute_mean``` function
from scipy.ndimage.filters import *


def compute_mean(image, filter_size):
    '''
    For each pixel in the image image consider a window of size
    filter_size x filter_size around the pixel and compute the mean
    of this window.
    @return: image containing the mean for each pixel
    '''

    return uniform_filter(image, filter_size, mode='reflect')


def compute_variance(image, filter_size):
    '''
    For each pixel in the image image consider a window of size
    filter_size x filter_size around the pixel and compute the variance
    of this window.
    @return: image containing the variance (\\sigma^2) for each pixel
    '''
    mean = compute_mean(image, filter_size)

    sqr_mean = compute_mean(image ** 2, filter_size)

    win_var = sqr_mean - mean ** 2
    return win_var


def compute_a(I, p, mu, p_bar, variance, eps, filter_size):
    '''
    Compute the intermediate result 'a' as described in the task in Eq. 3.
    @return: image containing a_k for each pixel
    '''
    mean_I_p = compute_mean(I*p, filter_size)

    a = (mean_I_p - (mu*p_bar))/(variance+eps)
    return a


def compute_b(p_bar, a, mu):
    '''
    Compute the intermediate result 'b' as described in the task in Eq. 5.
    @return: image containing b_k for each pixel
    '''
    b = p_bar - (a*mu)
    return b


def compute_q(a_bar, b_bar, I):
    '''
    Compute the final filtered result 'q' as described in the task (Eq. 5).
    @return: filtered image
    '''
    q = a_bar*I + b_bar
    return q


def fnf_denoising(image_dark, image_flash, r, eps):
    '''
    Call your guided image filtering method with the right parameters to
    perform flash-no-flash denoising.
    '''
    output = np.zeros_like(image_dark)
    for c in range(output.shape[2]):
        output[:, :, c] = guided_image_filtering(image_dark[:, :, c], image_flash[:, :, c], r, eps)
    return output


def detail_enhancement(inp, c=50, r=500, eps=1e-4):
    '''
    Call your guided image filtering method with the right parameters and
    implement Eq. 13 of the assignment sheet.
    '''
    output = np.zeros_like(inp)
    for color in range(inp.shape[2]):
        q = guided_image_filtering(inp[:, :, color], inp[:, :, color], r, eps)
        output[:, :, color] = c * (inp[:, :, color] - q) + q
    return output



def guided_image_filtering(p, I, r=1, eps=1):
    filter_size = 2 * r + 1
    p_bar, mu = compute_mean(p, filter_size), compute_mean(I, filter_size)
    variance = compute_variance(I, filter_size)
    a = compute_a(p, I, p_bar, mu, variance, eps, filter_size)
    b = compute_b(p_bar, a, mu)
    a_bar, b_bar = compute_mean(a, filter_size), compute_mean(b, filter_size)
    return compute_q(a_bar, b_bar, I)


if __name__ == "__main__":
    r = 100
    eps = 1e-2
    input = imageio.imread('./imgs/detail_enhancement/input.png') / 255.
    out = detail_enhancement(input, r, eps)
    imageio.imwrite(
         f'./imgs/detail_enhancement/out_r_{r}_eps_{eps:.4f}.png',
         out
     )

    '''r_values = [10, 100, 250, 500, 1000]
    eps_values = [1e-9, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    input_flash = imageio.imread('./imgs/fnf/flash.jpeg') / 255.
    input_no_flash = imageio.imread('./imgs/fnf/no_flash.jpeg') / 255.
    print(f'FnF: Image size is {input_flash.shape}')
    for r in r_values:
        out = fnf_denoising(input_no_flash, input_flash, r, eps_values[2])
        imageio.imwrite(
            f'./imgs/fnf/output_r_{r}.png',
            (out*255).astype(np.uint8)
        )

    for eps in eps_values:
        out = fnf_denoising(input_no_flash, input_flash, r_values[4], eps)
        imageio.imwrite(
            f'./imgs/fnf/output_eps_{eps}.png',
            (out*255).astype(np.uint8)
        )'''