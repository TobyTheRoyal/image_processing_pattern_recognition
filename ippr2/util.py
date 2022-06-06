import torch as th
import os
import torch.nn.functional as F
import imageio
import dct


def patch2image(batches, img_dims, size=(2, 2), stride=(1, 1)):
    tmp = batches[None].permute(0, 2, 1)
    return th.squeeze(F.fold(tmp, img_dims, size, 1, 0, stride))


def image2patch(image, size=(2, 2), stride=(1, 1)):
    return image.unfold(
        0, size[0], stride[0]
    ).unfold(
        1, size[1], stride[1]
    ).contiguous().view(-1, size[0] * size[1])


def get_train_data(p=2, N=int(2e6)):
    base_path = './train'
    imgs = [
        th.from_numpy(
            imageio.imread(os.path.join(base_path, f)) / 255
        ).mean(-1)[50:-50, 50:-50]
        for f in os.listdir(base_path) if f.endswith('.jpg')
    ]
    D = dct.mat2d(p)[1:]
    return th.cat([image2patch(img, (p, p)) @ D.T for img in imgs])[:N]


def psnr(a, b):
    mse = th.mean((a - b) ** 2)
    return 20 * th.log10(1 / th.sqrt(mse))
