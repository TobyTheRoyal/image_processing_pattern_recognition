import torch as th
import imageio
import matplotlib.pyplot as plt
import dct
import k_means as km
import math
import util


def mean_shift(e, n, c, w, s, h, D, prefactor):
    '''
    Implement the modified mean shift algorithm (Eq. 21)
    e, n ... shape (153600, d)
    c ... shape (5000, d-1)
    w ... shape (5000)
    D ... shape (d-1, d)
    prefactor ... shape (d, d)
    '''

    w_tilde = w * th.exp(-th.cdist(e @ D.T, c)**2 / (2 * h**2))  # shape (153600, 5000)
    return ((((w_tilde @ c) @ D).T / (th.sum(w_tilde, dim=1) * h**2)).T + n / s**2) @ prefactor


def kde(x, c, h, w=None):
    '''
    Implement the KDE of the prior (Eq. 7)
    '''
    N_c = c.size()[0]
    d = x.size()[1]
    return (th.exp(-th.cdist(x, c)**2 / (2*h**2)) @ w.type(th.float64)) / ((2*math.pi)**(d/2) * N_c * h**d)


def _main():
    iter = 25
    psnr = th.zeros(iter)
    device = th.device('cpu:0')
    sigma_n = 25 / 255
    h = 0.07
    patch_size = 4
    Nc = 5000
    kh, kw = patch_size, patch_size

    clean_I = th.mean(
        th.from_numpy(imageio.imread('./input.jpg') / 255), -1
    ).to(device)

    noisy_I = clean_I + clean_I.new_empty(clean_I.shape).normal_(0, sigma_n)
    noisy = util.image2patch(noisy_I.clone(), (kh, kw))
    imageio.imwrite(f'./out/input.png', th.clamp(noisy_I, 0, 1).cpu().numpy())

    estimate = noisy.clone()

    dct_bank, bank_weights = km.get_data(Nc, patch_size, device)

    # We transform the "one-image" to patches and back to get the proper
    # averaging factor to combine the patches
    div = util.patch2image(
        util.image2patch(clean_I.new_ones(clean_I.shape), (kh, kw)),
        clean_I.shape,
        (kh, kw)
    )

    # We introduce a degree of freedom here since the optimal sigma depends
    # on the choice of h
    sigma = sigma_n * 4

    # Here we discard the first row since this corresponds to the mean
    D = dct.mat2d(patch_size)[1:].to(device)

    prefactor = th.inverse(D.transpose(0, 1) @ D / h**2 + th.eye(D.size()[1]) / sigma**2)  # Calculate the prefactor (inverse mat in Eq. 21)

    # for trajectory
    ind = th.randint(0, estimate.size()[0] - 1, (10,)).tolist()
    traj = th.zeros((iter + 1, len(ind), patch_size**2))
    traj[0, :, :] = estimate[ind, :]

    plt.ion()
    f, a = plt.subplots(1, 3)
    a[1].imshow(clean_I.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    a[2].imshow(noisy_I.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    for i in range(iter):
        estimate = mean_shift(
            estimate, noisy,
            dct_bank, bank_weights,
            sigma, h, D, prefactor
        )

        traj[i + 1, :, :] = estimate[ind, :]
        # Transform to image and back to patches, essentially averaging
        # the intersection of the patches
        estimate_I = util.patch2image(estimate, clean_I.shape, (kh, kw)) / div
        estimate = util.image2patch(estimate_I, (kh, kw))


        psnr[i] = util.psnr(estimate_I, clean_I)
        a[0].imshow(estimate_I.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.pause(0.01)
        if psnr[i] < psnr[i-1]:
            print(f'# of iterations: {i}'
                  f'PSNR = {psnr[i]}')
            break

    estimate = util.patch2image(estimate, clean_I.shape, (kh, kw)) / div
    imageio.imwrite(f'./out/denoised_{Nc}_{patch_size**2}.png', th.clamp(estimate, 0, 1).cpu().numpy())
    th.save(traj, f'traj_{Nc}_{patch_size**2}.pth')

    plt.figure()
    plt.plot(psnr.numpy())
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('number of iterations')
    plt.ylabel('PSNR')
    plt.title(fr'PSNR over iterations for $d = {patch_size**2}$, $N_c = {Nc}$')
    plt.savefig(f'./out/psnr_{Nc}_{patch_size**2}.png', dpi=300)




if __name__ == '__main__':
    with th.no_grad():
        _main()
