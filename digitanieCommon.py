import numpy
import torch
import skimage


def numpyTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


def confusion(y, z):
    cm = torch.zeros(2, 2).cuda()
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cm[a][b] = ((z == a).float() * (y == b).float()).sum()

    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return cm, accu, iou0 + iou1


def confusionInstance(y, z):
    assert len(y.shape) == 2 and len(z.shape) == 2

    vtmap, nbVT = skimage.measure.label(y, return_num=True)
    predmap, nbPRED = skimage.measure.label(z, return_num=True)
    vtmap = torch.Tensor(vtmap).cuda() - 1
    predmap = torch.Tensor(predmap).cuda() - 1

    # chatGPT acceleration d'une double boucle for
    vt = vtmap.unsqueeze(0) == torch.arange(nbVT).cuda().unsqueeze(0).unsqueeze(0)
    pred = predmap.unsqueeze(0) == torch.arange(nbPRED).cuda().unsqueeze(0).unsqueeze(0)
    vt, pred = vt.flatten(1).unsqueeze(0), pred.flatten(1).unsqueeze(1)
    I = pred.float() * vt.float()
    U = pred.float() + vt.float() - I
    I, U = I.sum(dim=2), U.sum(dim=2)
    IoU = I / (U + 0.001)

    # suppression des blobs qui débordent sur 2 batiments
    for i in range(nbPRED):
        if (IoU[i] != 0).float().sum() > 1:
            IoU[i] = 0

    # matching
    L = [(-val, i, j) for (i, j), val in numpy.ndenumerate(IoU.cpu().numpy())]
    L = sorted(L)

    I, J, G = set(), set(), []
    for v, i, j in L:
        if v < 0.05:
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
    visu += tmp.unsequeeze(0) * 255
    tmp = (flag[0] == 1).float() * (flag[2] == 1).float()
    visu[1] += 255 * tmp
    tmp = (z == 0).float() * (flag[2] == 1).float()
    visu[2] += 255 * tmp
    tmp = (flag[1] == 1).float() * (flag[3] == 1).float()
    visu[0] += 255 * tmp
    visu[1] *= 1 - tmp
    visu[2] *= 1 - tmp
    visu = torch.clamp(visu, 0, 255)

    recall = len(G) / (nbVT + 0.00001)
    precision = len(G) / (nbPRED + 0.00001)
    gscore = recall * precision
    return gscore, recall, precision, visu, G


import queue
import threading
import PIL
from PIL import Image
import rasterio


class CropExtractor(threading.Thread):
    def __init__(self, paths, flag, tile=256):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = 500
        self.tilesize = tile

        pathdata, pathvt, suffixvt = paths
        self.pathdata = pathdata
        self.pathvt = pathvt
        self.suffixvt = suffixvt
        assert flag in ["even", "odd", "all"]
        self.flag = flag

        if self.flag == "all":
            self.NB = 10
        else:
            self.NB = 5

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        if self.flag == "odd":
            i = i * 2 + 1
        if self.flag == "even":
            i = i * 2

        with rasterio.open(self.pathdata + str(i) + ".tif") as src:
            r = minmax01(src.read(1))
            g = minmax01(src.read(2))
            b = minmax01(src.read(3))
            x = numpy.stack([r, g, b], axis=-1)

        y = PIL.Image.open(self.pathvt + str(i) + self.suffixvt).convert("RGB").copy()
        y = numpy.asarray(y)
        y = numpy.uint8((y[:, :, 0] == 250) * (y[:, :, 1] == 50) * (y[:, :, 2] == 50))

        if torchformat:
            return numpyTOtorch(x), smooth(torch.Tensor(y))
        else:
            return x, y

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize
        debug = True

        while True:
            for i in range(self.NB):
                image, label = self.getImageAndLabel(i)

                ntile = 50
                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]

                    if numpy.sum(numpy.int64(mask != 0)) == 0:
                        continue
                    if numpy.sum(numpy.int64(mask == 0)) == 0:
                        continue

                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = numpyTOtorch(x), smooth(torch.Tensor(y))
                    self.q.put((x, y), block=True)


class DIGITANIE:
    def __init__(self, root, infos, flag, tile=256):
        self.run = False
        self.tilesize = tile
        self.cities = [city for city in infos]
        self.root = root
        self.NBC = len(self.cities)

        self.allimages = []
        self.data = {}
        for city in self.cities:
            pathdata = root + city + infos[city]["pathdata"]
            pathvt = root + city + "/COS9/" + city + "_"
            paths = pathdata, pathvt, infos[city]["suffixvt"]
            self.data[city] = CropExtractor(paths, flag, tile=tile)
            self.allimages.extend([(city, i) for i in range(self.data[city].NB)])
        self.NB = len(self.allimages)

    def getImageAndLabel(self, i, torchformat=False):
        city, j = self.allimages[i]
        return self.data[city].getImageAndLabel(j, torchformat=torchformat)

    def start(self):
        if not self.run:
            self.run = True
            for city in self.cities:
                self.data[city].start()

    def getBatch(self, batchsize):
        assert self.run
        batchchoice = (torch.rand(batchsize) * self.NBC).long()

        x = torch.zeros(batchsize, 3, self.tilesize, self.tilesize)
        y = torch.zeros(batchsize, self.tilesize, self.tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.data[self.cities[batchchoice[i]]].getCrop()
        return x, y


def getDIGITANIE(flag, root="/scratchf/AI4GEO/DIGITANIE/", tile=256):
    assert flag in ["odd", "even", "all"]

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

    return DIGITANIE(root, infos, flag, tile=tile)


import torchvision
import samAsInput


class GlobalLocal(torch.nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()
        self.backbone = torchvision.models.efficientnet_v2_l(weights="DEFAULT").features

        self.compress = torch.nn.Conv2d(1280, 32, kernel_size=1)

        with torch.no_grad():
            old = self.backend[0][0].weight.data.clone() / 2
            self.backend[0][0] = torch.nn.Conv2d(
                6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            tmp = torch.cat([old, old], dim=1)
            self.backend[0][0].weight = torch.nn.Parameter(tmp)

    def forward(self, x):
        _, _, h, w = x.shape
        x = ((x / 255) - 0.5) / 0.25
        x = self.backend(x)
        x = self.classif(x)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear")
        return x

    def forwardglobal(self, x):
        x = 2 * x - 1
        x = torch.nn.functional.interpolate(
            x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bilinear"
        )
        return torch.nn.functional.leaky_relu(self.backbone(x))

    def forwardlocal(self, x, f):
        z = self.local1(x)
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local2(z))
        z = torch.cat([z, x, z * f, f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local3(z))

        zz = torch.nn.functional.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        zz = torch.nn.functional.relu(100 * z - 99 * zz)

        z = torch.cat([z, zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local4(z))
        z = torch.cat([z, z * zz, z * f, zz * f], dim=1)
        z = torch.nn.functional.leaky_relu(self.local5(z))
        return self.classifhigh(z)

    def forward(self, x, mode="normal"):
        assert mode in ["normal", "globalonly", "nofinetuning"]

        if mode != "normal":
            with torch.no_grad():
                f = self.forwardglobal(x)
        else:
            f = self.forwardglobal(x)

        z = self.classiflow(f)
        z = torch.nn.functional.interpolate(
            z, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        if mode == "globalonly":
            return z

        f = torch.nn.functional.leaky_relu(self.compress(f))
        f = torch.nn.functional.interpolate(
            f, size=(x.shape[2], x.shape[3]), mode="bilinear"
        )
        return self.forwardlocal(x, f) + z * 0.1


def mapfiltered(spatialmap, setofvalue):
    def myfunction(i):
        return int(int(i) in setofvalue)

    myfunctionVector = numpy.vectorize(myfunction)
    return myfunctionVector(spatialmap)


def sortmap(spatialmap):
    tmp = torch.Tensor(spatialmap)
    nb = int(tmp.flatten().max())
    tmp = sorted([(-(tmp == i).float().sum(), i) for i in range(1, nb + 1)])
    valuemap = {}
    valuemap[0] = 0
    for i, (k, j) in enumerate(tmp):
        valuemap[j] = i + 1

    def myfunction(i):
        return int(valuemap[int(i)])

    myfunctionVector = numpy.vectorize(myfunction)
    return myfunctionVector(spatialmap)
