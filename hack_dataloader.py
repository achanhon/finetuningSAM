import os
import PIL
from PIL import Image
import numpy
import torch
import random
import queue
import threading


def perf(cm):
    accu = 100.0 * (cm[0][0] + cm[1][1]) / (torch.sum(cm) + 1)
    iou0 = 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1] + 1)
    iou1 = 50.0 * cm[1][1] / (cm[1][1] + cm[1][0] + cm[0][1] + 1)
    return torch.Tensor((iou0 + iou1, accu))


def symetrie(x, y, ijk):
    i, j, k = ijk[0], ijk[1], ijk[2]
    if i == 1:
        x, y = numpy.transpose(x, axes=(1, 0, 2)), numpy.transpose(y, axes=(1, 0))
    if j == 1:
        x, y = numpy.flip(x, axis=1), numpy.flip(y, axis=1)
    if k == 1:
        x, y = numpy.flip(x, axis=0), numpy.flip(y, axis=0)
    return x.copy(), y.copy()


def pilTOtorch(x):
    return torch.Tensor(numpy.transpose(x, axes=(2, 0, 1)))


def torchTOpil(x):
    return numpy.transpose(x.cpu().numpy(), axes=(1, 2, 0))


class CropExtractor(threading.Thread):
    def __init__(self, path, maxsize=50, tilesize=256):
        threading.Thread.__init__(self)
        self.isrunning = False
        self.maxsize = maxsize

        self.path = path
        self.NB = 0
        self.tilesize = tilesize
        while os.path.exists(self.path + str(self.NB) + "_x.png"):
            self.NB += 1

        if self.NB == 0:
            print("wrong path", self.path)
            quit()

    def getImageAndLabel(self, i, torchformat=False):
        assert i < self.NB

        image = PIL.Image.open(self.path + str(i) + "_x.png").convert("RGB").copy()
        image = numpy.uint8(numpy.asarray(image))

        label = PIL.Image.open(self.path + str(i) + "_y.png").convert("L").copy()
        label = numpy.uint8(numpy.asarray(label))
        label = numpy.uint8(label != 0)

        if torchformat:
            return pilTOtorch(image), torch.Tensor(label)
        else:
            return image, label

    ###############################################################

    def getCrop(self):
        assert self.isrunning
        return self.q.get(block=True)

    def getBatch(self, batchsize=2):
        tilesize = self.tilesize
        x = torch.zeros(batchsize, 3, self.tilesize, tilesize)
        y = torch.zeros(batchsize, tilesize, tilesize)
        for i in range(batchsize):
            x[i], y[i] = self.getCrop()
        return x, y.long()

    def run(self):
        self.isrunning = True
        self.q = queue.Queue(maxsize=self.maxsize)
        tilesize = self.tilesize

        while True:
            I = [i for i in range(self.NB)]
            random.shuffle(I)
            for i in I:
                image, label = self.getImageAndLabel(i, torchformat=False)

                ntile = 10
                RC = numpy.random.rand(ntile, 2)
                flag = numpy.random.randint(0, 2, size=(ntile, 3))
                for j in range(ntile):
                    r = int(RC[j][0] * (image.shape[0] - tilesize - 2))
                    c = int(RC[j][1] * (image.shape[1] - tilesize - 2))
                    im = image[r : r + tilesize, c : c + tilesize, :]
                    mask = label[r : r + tilesize, c : c + tilesize]
                    x, y = symetrie(im.copy(), mask.copy(), flag[j])
                    x, y = pilTOtorch(x), torch.Tensor(y)
                    self.q.put((x, y), block=True)
