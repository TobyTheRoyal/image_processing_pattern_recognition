# Do not import any other modules
import imageio
import matplotlib.pyplot as plt
import numpy as np
from guided_filtering import guided_image_filtering
# You may use **one** filter of the following module to implement the
# ```dark_channel``` function
from scipy.ndimage import *


def dark_channel(im, patch_size):
    '''
    Extract the "dark channel" from the image `im`, using a patch size
    of `patch_size` pixels (Eq. 7).
    '''

    dark = minimum_filter(im, size=patch_size, mode='reflect')
    return np.amin(dark, axis=2)


def atmosphere_color(im, dark):
    '''
    Get the 0.1% brightest pixels in the dark channel, and get the color of the
    atmosphere from the brightest pixels at these locations in the original
    image.
    '''

    N = int(dark.shape[0] * dark.shape[1] * 0.001)
    idx = np.argpartition(dark, -N, axis=None)[-N:]
    indices = idx[np.argsort((-im.flatten())[idx])]
    brightness = np.mean(im, axis=2)
    idx = np.argpartition(brightness.flatten()[indices], -int(N / 100))[-int(N / 100):]
    indices = idx[np.argsort((-brightness.flatten()[indices].flatten())[idx])]
    y,x = np.unravel_index(indices, shape=dark.shape)
    return np.mean(im[y,x,:], axis=0)


def transmission(im, atm, patch_size, omega=0.95):
    '''
    Calculate the transmission map according to Eq. 11.
    '''

    return 1 - omega * np.amin(minimum_filter(im / atm[np.newaxis, np.newaxis, :], size=patch_size), axis=2)


def refine_transmission(im, t, r=1, eps=1):
    '''
    Refine the transmission estimate using the guided image filter.
    '''

    return guided_image_filtering(t, np.mean(im, axis=2), r, eps)


def restore(im, t, atm, t_0=0.1):
    '''
    Estimate the scene radiance (Eq. 12).
    '''

    t[t < t_0] = t_0
    return (im-atm[np.newaxis,np.newaxis,:]) / t[:,:,np.newaxis] + atm[np.newaxis, np.newaxis,:]


if __name__ == '__main__':
    input = imageio.imread('./imgs/dehazing/input.png') / 255.

    patch_size = 15

    dark = dark_channel(input, patch_size)
    atmosphere = atmosphere_color(input, dark)
    t_estimate = transmission(input, atmosphere, patch_size)
    t_refined = refine_transmission(input, t_estimate, r=25, eps=1e-9)
    restored = restore(input, t_refined, atmosphere, t_0=0.1)
    restored_bad = restore(input, t_estimate, atmosphere, t_0=0.1)

    imageio.imsave('./imgs/dehazing/restored.png', np.clip(restored, 0, 1))
    imageio.imsave('./imgs/dehazing/te.png', np.clip(t_estimate, 0, 1))
    imageio.imsave('./imgs/dehazing/t.png', np.clip(t_refined, 0, 1))
    imageio.imsave('./imgs/dehazing/dark.png', np.clip(dark, 0, 1))
    imageio.imsave('./imgs/dehazing/restored_bad.png', np.clip(restored_bad, 0, 1))

    input = imageio.imread('./imgs/dehazing/input_custom6.jpeg') / 255.

    # patch_size = 50
    #
    # dark = dark_channel(input, patch_size)
    # atmosphere = atmosphere_color(input, dark)
    # t_estimate = transmission(input, atmosphere, patch_size)
    # t_refined = refine_transmission(input, t_estimate, r=25, eps=1e-9)
    # restored = restore(input, t_refined, atmosphere, t_0=0.1)
    # restored_bad = restore(input, t_estimate, atmosphere, t_0=0.1)
    #
    # imageio.imsave('./imgs/dehazing/restored_custom6.png', np.clip(restored, 0, 1))
    # imageio.imsave('./imgs/dehazing/te_custom6.png', np.clip(t_estimate, 0, 1))
    # imageio.imsave('./imgs/dehazing/t_custom6.png', np.clip(t_refined, 0, 1))
    # imageio.imsave('./imgs/dehazing/dark_custom6.png', np.clip(dark, 0, 1))
    # imageio.imsave('./imgs/dehazing/restored_bad_custom6.png', np.clip(restored_bad, 0, 1))
