import numpy
import torch
import skimage


def confusion(y, z):
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = ((z == a).float() * (y == b).float()).sum()
    return cm


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return accu, iou0 + iou1


def confusionInstance(y, z):
    assert len(y.shape) == 2 and len(z.shape) == 2

    y_, z_ = y.cpu().numpy(), z.cpu().numpy()
    vtmap, nbVT = skimage.measure.label(y_, return_num=True)
    predmap, nbPRED = skimage.measure.label(z_, return_num=True)
    vtmap = torch.Tensor(vtmap).cuda() - 1
    predmap = torch.Tensor(predmap).cuda() - 1

    # calcul des IoU
    IoU = torch.zeros(nbPRED, nbVT)
    for a in range(nbPRED):
        pred = (predmap == a).float()
        for b in range(nbVT):
            vt = (vtmap == b).float()
            I = pred * vt
            U = pred + vt - I
            IoU[a][b] = I.sum() / (U.sum() + 0.001)

    # suppression des blobs qui débordent sur 2 batiments
    for i in range(nbPRED):
        if (IoU[i] != 0).float().sum() > 1:
            IoU[i] = 0

    # matching
    L = [(-val, i, j) for (i, j), val in numpy.ndenumerate(IoU.cpu().numpy())]
    L = sorted(L)

    I, J, G = set(), set(), []
    for v, i, j in L:
        if (-v) < 0.05:
            break
        if i in I or j in J:
            continue
        G.append((i, j))
        I.add(i)
        J.add(j)

    # vert = parfait - pixel d'un building capturé recouvert
    # rouge = pire - pixel d'un building non capturé mais recouvert
    # bleu = pixel d'un building capturé non recouvert
    # blanc = pixel d'un building non capturé
    # jaune = pred qui déborde
    # orange = hallucination

    flag = torch.zeros(4, y.shape[0], y.shape[1]).cuda()
    for i in range(nbPRED):
        if i in I:
            flag[0] += (predmap == i).float()
        else:
            flag[1] += (predmap == i).float()
    for j in range(nbVT):
        if j in J:
            flag[2] += (vtmap == j).float()
        else:
            flag[3] += (vtmap == j).float()

    visu = torch.zeros(3, y.shape[0], y.shape[1]).cuda()
    tmp = (flag[1] == 1).float() * (y == 0).float()
    visu[0] += 255 * tmp
    visu[1] += 165 * tmp
    tmp = (flag[0] == 1).float() * (y == 0).float()
    visu[0] += 255 * tmp
    visu[1] += 255 * tmp
    tmp = (z == 0).float() * (flag[3] == 1).float()
    visu[0] += 255 * tmp
    visu[1] += 255 * tmp
    visu[2] += 255 * tmp
    tmp = (flag[0] == 1).float() * (flag[2] == 1).float()
    visu[1] += 255 * tmp
    tmp = (z == 0).float() * (flag[2] == 1).float()
    visu[2] += 255 * tmp
    tmp = (flag[1] == 1).float() * (flag[3] == 1).float()
    visu[0] += 255 * tmp
    visu[1] *= 1 - tmp
    visu[2] *= 1 - tmp
    visu = torch.clamp(visu, 0, 255)
    return len(G), nbVT, nbPRED, visu


def perfI(G, nbVT, nbPRED):
    recall = G / (nbVT + 0.00001)
    precision = G / (nbPRED + 0.00001)
    gscore = recall * precision
    return gscore, recall, precision


import torchvision
import samAsInput


class Deeplab(torch.nn.Module):
    def __init__(self, outS=2):
        super(Deeplab, self).__init__()
        self.backend = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DEFAULT"
        )
        self.backend.classifier[4] = torch.nn.Conv2d(256, outS, kernel_size=1)

    def forward(self, x):
        return self.backend(x)["out"]


class FUSION(torch.nn.Module):
    def __init__(self):
        super(FUSION, self).__init__()
        self.sam = samAsInput.SAMasInput()

        self.net = Deeplab(outS=16)

        self.c1 = torch.nn.Conv2d(50, 50, kernel_size=1)
        self.c2 = torch.nn.Conv2d(50, 50, kernel_size=7, padding=3)
        self.c3 = torch.nn.Conv2d(50, 50, kernel_size=1)
        self.c4 = torch.nn.Conv2d(50, 34, kernel_size=5, padding=2)
        self.c5 = torch.nn.Conv2d(50, 2, kernel_size=1)

        with torch.no_grad():
            old = self.net.backend.backbone.conv1.weight.data.clone() / 2
            self.net.backend.backbone.conv1 = torch.nn.Conv2d(
                6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            tmp = torch.cat([old, old], dim=1)
            self.net.backend.backbone.conv1.weight = torch.nn.Parameter(tmp)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        B, _, h, w = x.shape

        _, mask = self.sam.applySAMmultiple(x)

        tmpm = x.mean(dim=1).unsqueeze(1)
        tmp1 = torch.ones(B, 1, h, w).cuda() * 255
        x = torch.cat([x, mask * 255, tmpm, tmp1], dim=1)

        x = ((x / 255) - 0.5) / 0.25
        x = self.net(x)

        xx = 10 * x - 9 * torch.nn.functional.max_pool2d(
            x, kernel_size=3, padding=1, stride=1
        )
        xx = torch.cat([x, xx, x * (1 - mask), mask, (1 - mask)], dim=1)

        xx = self.lrelu(self.c1(xx))
        xx = self.lrelu(self.c2(xx))
        xx = self.lrelu(self.c3(xx))
        xx = self.lrelu(self.c4(xx))
        x = torch.cat([x, xx], dim=1)
        return self.c5(x)


import queue
import threading
import PIL
from PIL import Image
import rasterio
import random


class CropExtractorDigitanie(threading.Thread):
    def __init__(self, paths, flag, tile):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 10
        self.tilesize = tile

        assert flag in ["even", "odd", "all"]
        self.flag = flag

        if self.flag == "even":
            paths = paths[::2]
        if self.flag == "odd":
            paths = paths[1::2]
        self.paths = paths

    def getImageAndLabel(self, i, torchformat=True):
        assert i < len(self.paths)

        with rasterio.open(self.paths[i][0]) as src:
            r = numpy.clip(src.read(1) * 2, 0, 1)
            g = numpy.clip(src.read(2) * 2, 0, 1)
            b = numpy.clip(src.read(3) * 2, 0, 1)
            x = numpy.stack([r, g, b], axis=0) * 255

        y = PIL.Image.open(self.paths[i][1]).convert("RGB").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :, 0] == 250) * (y[:, :, 1] == 50) * (y[:, :, 2] == 50))

        if y.shape != (2048, 2048):
            y, x = y[0:2048, 0:2048], x[:, 0:2048, 0:2048]

        if torchformat:
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=5):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def symetrie(self, x, y):
        if random.random() > 0.5:
            x[0] = numpy.transpose(x[0], axes=(1, 0))
            x[1] = numpy.transpose(x[1], axes=(1, 0))
            x[2] = numpy.transpose(x[2], axes=(1, 0))
            y = numpy.transpose(y, axes=(1, 0))
        if random.random() > 0.5:
            x[0] = numpy.flip(x[0], axis=1)
            x[1] = numpy.flip(x[1], axis=1)
            x[2] = numpy.flip(x[2], axis=1)
            y = numpy.flip(y, axis=1)
        if random.random() > 0.5:
            x[0] = numpy.flip(x[0], axis=0)
            x[1] = numpy.flip(x[1], axis=0)
            x[2] = numpy.flip(x[2], axis=0)
            y = numpy.flip(y, axis=0)
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize
        I = list(range(len(self.paths)))

        while True:
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                r = int(random.random() * 1500)
                c = int(random.random() * 1500)
                im = image[:, r : r + tilesize, c : c + tilesize]
                mask = label[r : r + tilesize, c : c + tilesize]

                if numpy.sum(numpy.int64(mask != 0)) == 0:
                    continue
                if numpy.sum(numpy.int64(mask == 0)) == 0:
                    continue

                x, y = self.symetrie(im.copy(), mask.copy())
                x, y = torch.Tensor(x.copy()), torch.Tensor(y.copy())
                self.q.put((x, y), block=True)


def getDIGITANIE(flag, root="/scratchf/AI4GEO/DIGITANIE/", tile=512):
    infos = {}
    infos["Arcachon"] = {"pathdata": "/Arcachon_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Biarritz"] = {"pathdata": "/Biarritz_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Brisbane"] = {"pathdata": "/Brisbane_EPSG32756_", "suffixvt": "-v4.tif"}
    infos["Can-Tho"] = {"pathdata": "/Can-Tho_EPSG32648_", "suffixvt": "-v4.tif"}
    infos["Helsinki"] = {"pathdata": "/Helsinki_EPSG32635_", "suffixvt": "-v4.tif"}
    infos["Lagos"] = {"pathdata": "/Lagos_EPSG32631_", "suffixvt": "_mask.tif"}
    infos["Maros"] = {"pathdata": "/Maros_EPSG32750_", "suffixvt": "-v4.tif"}
    infos["Montpellier"] = {"pathdata": "/Montpellier_EPSG2154_", "suffixvt": "-v4.tif"}
    infos["Munich"] = {"pathdata": "/Munich_EPSG32632_", "suffixvt": "-v4.tif"}
    infos["Nantes"] = {"pathdata": "/Nantes_EPSG32630_", "suffixvt": "-v4.tif"}
    infos["Paris"] = {"pathdata": "/Paris_EPSG2154_", "suffixvt": "-v4.tif"}
    infos["Port-Elisabeth"] = {
        "pathdata": "/Port-Elisabeth_EPSG32735_",
        "suffixvt": "_mask.tif",
    }
    infos["Shanghai"] = {"pathdata": "/Shanghai_EPSG32651_", "suffixvt": "-v4.tif"}
    infos["Strasbourg"] = {"pathdata": "/Strasbourg_EPSG32632_", "suffixvt": "-v4.tif"}
    infos["Tianjin"] = {"pathdata": "/Tianjin_32650_", "suffixvt": "-v4.tif"}
    infos["Toulouse"] = {"pathdata": "/Toulouse_EPSG32631_", "suffixvt": "-v4.tif"}

    paths = []
    for city in infos:
        for i in range(10):
            x = root + city + infos[city]["pathdata"] + str(i) + ".tif"
            y = root + city + "/COS9/" + city + "_" + str(i) + infos[city]["suffixvt"]
            paths.append((x, y))

    return CropExtractorDigitanie(paths, flag, tile=tile)


if __name__ == "__main__":
    import os

    os.system("rm -r build")
    os.system("mkdir build")

    if False:
        net = GlobalLocal()
        net.eval()
        net.cuda()
        tmp = torch.rand(2, 3, 512, 512).cuda()
        tmp[:, :, 200:300, 200:300] = 1
        print(net(tmp).shape)
        quit()

    if False:
        dataset = getDIGITANIE("all")
        dataset.start()
        x, _ = dataset.getBatch()
        torchvision.utils.save_image(x / 255, "build/lol.png")
        os._exit(0)

    if False:
        dataset = getDIGITANIE("all")
        dataset.start()
        x, _ = dataset.getBatch()

        net = GlobalLocal()
        net.eval()
        net.cuda()

        _, border = net.sam.applySAMmultiple(x.cuda())
        torchvision.utils.save_image(x / 255, "build/x.png")
        torchvision.utils.save_image(border, "build/z.png")
        os._exit(0)

    os.system("/d/achanhon/miniconda3/bin/python -u digitanieTrain.py")
    os.system("/d/achanhon/miniconda3/bin/python -u digitanieTest.py")
