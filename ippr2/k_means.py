from sklearn.cluster import MiniBatchKMeans as KMeans
import os
import torch as th
import util


def k_means_faiss(n_c, p, N=int(8e6)):
    import faiss
    x = util.get_train_data(p, N).numpy().astype('float32')
    d = x.shape[1]
    clus = faiss.Clustering(d, n_c)
    clus.max_points_per_centroid = 1000000

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0

    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d, cfg)
    clus.train(x, index)
    c = th.from_numpy(faiss.vector_float_to_array(clus.centroids).reshape(n_c, d)).to(th.float64)
    _, i = index.search(x, 1)
    w = th.bincount(th.squeeze(th.from_numpy(i))) / x.shape[0]

    _save(c, w, n_c, p)
    return c, w


def k_means_th(n_c, p, N=int(2e6)):
    x = util.get_train_data(p, N).to('cpu:0')
    c = x[:n_c].clone()
    while True:
        c_old = c.clone()
        cl = th.cdist(x, c).argmin(dim=1)
        div = th.bincount(cl)[..., None]
        c.zero_().index_add_(0, cl, x).div_(div)
        if th.allclose(c_old, c):
            break

    w = th.bincount(cl) / x.shape[0]
    _save(c, w, n_c, p)
    return c, w


def k_means_sk(n_c=500, p=15, N=int(2e6)):
    dct_patches = util.get_train_data(p)
    km = KMeans(n_clusters=n_c).fit(dct_patches.numpy())
    centers = th.tensor(km.cluster_centers_)
    weights = th.bincount(th.tensor(km.labels_)) / float(dct_patches.shape[0])
    _save(centers, weights, n_c, p)
    return centers, weights


def get_data(n_c, p, mode='th', device='cpu:0', N=int(2e6)):
    p_centers, p_weights = _filenames(n_c, p)
    if os.path.isfile(p_centers) and os.path.isfile(p_weights):
        c, w = th.load(p_centers), th.load(p_weights)
    else:
        if mode == 'faiss':
            kmeans_impl = k_means_faiss
        elif mode == 'th':
            kmeans_impl = k_means_th
        else:
            kmeans_impl = k_means_sk
        c, w = kmeans_impl(n_c, p, N)

    return c.to(device), w.to(device)


def _filenames(n_c, p):
    base = './km'
    return (
        os.path.join(base, f'centers_{n_c}_p{p}.pth'),
        os.path.join(base, f'weights_{n_c}_p{p}.pth'),
    )


def _save(c, w, n_c, p):
    p_centers, p_weights = _filenames(n_c, p)
    th.save(c, p_centers)
    th.save(w, p_weights)


if __name__ == '__main__':
    import time
    t = time.time()
    csk = k_means_faiss(n_c=5000, p=4)
    print(f'elapsed {time.time() - t}')
