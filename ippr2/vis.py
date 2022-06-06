import torch as th
import util
import mean_shift as ms
import matplotlib.pyplot as plt
import os
import k_means as km


base_path = 'kdes'
h = 0.05
# Number of discretization points along one dimension in DCT space
n_points = 50
n_c = 5000
h_km = 0.07

import dct
D = dct.mat2d(2)[1:].to('cpu')
traj = th.load('traj_5000_4.pth').transpose(0, 2)
traj = traj.reshape(4, -1)
traj_dct = D @ traj.type(th.float64)
traj_dct = traj_dct.reshape(3, 10, -1)


def calc_kde():
    device = th.device('cpu:0')
    p = 2
    dct_dims = p ** 2 - 1
    gx, gy, gz = th.meshgrid([th.linspace(-1, 1, n_points) for _ in range(3)])
    gx, gy, gz = gx.to(device), gy.to(device), gz.to(device)
    dct_all = util.get_train_data(p).to(device)
    grid = th.stack((gx, gy, gz), dim=-1).view(
        -1, dct_dims
    ).to(device).to(th.float64)
    kdes = []
    dct_km, w_km = km.get_data(n_c, p)
    # adjust the 500 such that the data still fits on your gpu
    for points in th.split(grid, 50):
        # kdes.append(ms.kde(points, dct_all, h))
        kdes.append(ms.kde(points, dct_km, h_km, w_km))
    kdes = th.stack(kdes).view(n_points, n_points, n_points)
    th.save(kdes, os.path.join(base_path, f'appr_h_{h_km}.pth'))


def visualize():
    gx, gy = th.meshgrid([th.linspace(-1, 1, n_points) for _ in range(2)])
    path_true = os.path.join(base_path, f'true_h_{h}.pth')
    path_appr = os.path.join(base_path, f'appr_h_{h_km}.pth')

    calc_kde()

    # Marginalization over the third DCT dimension
    kde_clean = th.load(path_true, map_location=th.device('cpu'))
    kde_appr = th.load(path_appr, map_location=th.device('cpu'))


    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')
    a.plot_surface(gx.numpy(), gy.numpy(), th.clamp_min(th.log(th.sum(kde_clean, dim=-1)), -30).numpy())
    a.set_zlim(-31, 1)
    a.set_xlabel('x-axis')
    a.set_ylabel('y-axis')
    a.set_zlabel(r'$p(x)$')
    plt.title(fr'The true KDE, $h = {h}$')
    plt.savefig(f'./out/clean.png', dpi=300)

    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')
    a.plot_surface(gx.numpy(), gy.numpy(), th.clamp_min(th.log(th.sum(kde_appr, dim=-1)), -30).numpy())
    a.set_zlim(-31, 1)
    a.set_xlabel('x-axis')
    a.set_ylabel('y-axis')
    a.set_zlabel(r'$\hat{p}(x)$')
    plt.title(fr'The KDE approximation, $N_\mathrm{{c}} = {n_c}$, $h_\mathrm{{km}} = {h_km}$')
    plt.savefig(f'./out/appr_{h_km}.png', dpi=300)

    for dim in range(3):
        f, a = plt.subplots(1, 1)
        a.imshow(th.clamp_min(th.log(th.sum(kde_clean, dim=dim)), -30).numpy(), extent=[-1,1,-1,1], vmin=-20, vmax=1.1)
        a.set_xlabel('x-axis')
        a.set_ylabel('y-axis')
        plt.title(f'True KDE, $N_\mathrm{{c}} = {n_c}$, $h = {h}$')
        for i in range(10):
            plt.plot(traj_dct[(dim - 2) % 3, i, :], traj_dct[(dim - 1) % 3, i, :], c=f'C{i}')
            plt.scatter(traj_dct[(dim - 2) % 3, i, 0], traj_dct[(dim - 1) % 3, i, 0], c=f'C{i}', marker='x')
        plt.savefig(f'./out/clean_proj{dim}.png', dpi=300)

        f, a = plt.subplots(1, 1)
        a.imshow(th.clamp_min(th.log(th.sum(kde_appr, dim=dim)), -30).numpy(), extent=[-1, 1, -1, 1], vmin=-20, vmax=1.1)
        a.set_xlabel('x-axis')
        a.set_ylabel('y-axis')
        plt.title(f'Approximated KDE, $N_\mathrm{{c}} = {n_c}$, $h_\mathrm{{km}} = {h_km}$')
        for i in range(10):
            plt.plot(traj_dct[(dim - 2) % 3, i, :], traj_dct[(dim - 1) % 3, i, :], c=f'C{i}')
            plt.scatter(traj_dct[(dim - 2) % 3, i, 0], traj_dct[(dim - 1) % 3, i, 0], c=f'C{i}', marker='x')
        plt.savefig(f'./out/appr_proj{dim}.png', dpi=300)


if __name__ == '__main__':
    visualize()
