import torch
import numpy
import rasterio
from functools import lru_cache


@lru_cache
def readSEN(path):
    return numpy.load(path)


class FLAIR2:
    def __init__(self, flag="test", root="/d/achanhon/FLAIR_2/"):
        assert flag in ["train", "val", "trainval"]
        self.root = root
        self.flag = flag
        self.paths = torch.load(root + "alltrainpaths.pth")

        tmp = sorted(self.paths.keys())
        if flag == "train":
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 != 0]
        if flag == "val":
            tmp = [k for (i, k) in enumerate(tmp) if i % 4 == 0]
        self.paths = {k: self.paths[k] for k in tmp}

    def get(self, k):
        assert k in self.paths
        with rasterio.open(self.root + self.paths[k]["image"]) as src:
            r = numpy.clip(src.read(1), 0, 255)
            g = numpy.clip(src.read(2), 0, 255)
            b = numpy.clip(src.read(3), 0, 255)
            i = numpy.clip(src.read(4), 0, 255)
            e = numpy.clip(src.read(5), 0, 255)
            x = numpy.stack([r, g, b, i, e], axis=0) * 255

        sentinel = readSEN(self.root + self.paths[k]["sen"])
        assert sentinel.shape[0:2] == (10, 20)
        row, col = self.paths[k]["coord"]
        sen = sentinel[:, :, row : row + 40, col : col + 40]
        sen = torch.Tensor(sen).flatten(0, 1)

        if self.flag != "test":
            with rasterio.open(self.root + self.paths[k]["label"]) as src:
                y = torch.Tensor(numpy.clip(src.read(1), 1, 13) - 1)
            return torch.Tensor(x), sen, y
        else:
            return torch.Tensor(x), sen


import segment_anything


class spatialSAM:
    def __init__(self):
        self.path = "model/sam_vit_b_01ec64.pth"
        self.sam = segment_anything.sam_model_registry["vit_b"](checkpoint=self.path)
        self.sam = self.sam.cuda()
        self.sam.eval()

        tmp = []
        for row in range(11, 504, 23):
            for col in range(11, 504, 23):
                tmp.append((row, col))
        self.magrille = torch.zeros(len(tmp), 1, 2).cuda()
        self.magrilleL = torch.zeros(len(tmp), 1).cuda()
        for i, (row, col) in enumerate(tmp):
            self.magrille[i][0][0] = row
            self.magrille[i][0][1] = col
            self.magrilleL[i][0] = i

        self.palette = torch.Tensor(
            [
                [255.0, 0.0, 0.0],
                [0.0, 255.0, 0.0],
                [0.0, 0.0, 255.0],
                [109.0, 98.0, 188.0],
                [230.0, 251.0, 148.0],
                [5.0, 18.0, 230.0],
                [163.0, 38.0, 214.0],
                [173.0, 142.0, 45.0],
                [95.0, 19.0, 120.0],
                [0.0, 0.0, 0.0],
            ]
        ).cuda()

    def applySAM_(self, x_):
        x = {}
        x["image"] = x_.cuda()
        x["original_size"] = (512, 512)
        x["point_coords"] = self.magrille
        x["point_labels"] = self.magrilleL

        masks = self.sam([x], False)[0]["masks"]

        # get the largest mask for each point
        masks, _ = masks.max(1)
        masks = (masks > 0).float()

        # erosion
        masks = 1 - torch.nn.functional.max_pool2d(
            1 - masks, kernel_size=3, stride=1, padding=1
        )

        # NMS
        tmp = [(masks[i].sum(), i) for i in range(masks.shape[0])]
        tmp = sorted(tmp)
        remove = []
        for i in range(len(tmp)):
            if tmp[i][0] == 0:
                remove.append(i)
                break
            if i in remove:
                continue
            for j in range(i, len(tmp)):
                if tmp[i][0] <= tmp[j][0] * 0.7:
                    break
                else:
                    I = masks[tmp[i][1]] * masks[tmp[j][1]]
                    U = masks[tmp[i][1]] + masks[tmp[j][1]] - I
                    IoU = I.sum() / (U.sum() + 0.1)
                    if IoU > 0.7:
                        remove.append(i)
                        break
        remove = set(remove)
        masks = masks[[i for i in range(len(tmp)) if i not in remove]]

        tmp = [(-masks[i].sum(), i) for i in range(masks.shape[0])]
        partition = torch.zeros(512, 512).cuda()
        for _, i in tmp:
            partition = partition * (1 - masks[i]) + masks[i] * (i + 1)
        nb = int(partition.flatten().max())

        pseudocolor = torch.zeros(x_.shape).cuda()
        for j in range(1, nb):
            pseudocolor[0] = (
                pseudocolor[0] * (1 - (partition == j).float())
                + self.palette[j % 9][0] * (partition == j).float()
            )
            pseudocolor[1] = (
                pseudocolor[1] * (1 - (partition == j).float())
                + self.palette[j % 9][0] * (partition == j).float()
            )
            pseudocolor[1] = (
                pseudocolor[2] * (1 - (partition == j).float())
                + self.palette[j % 9][0] * (partition == j).float()
            )

        return partition, pseudocolor

    def applySAM(self, x):
        with torch.no_grad():
            return self.applySAM_(x)


if __name__ == "__main__":
    import torchvision
    import random
    import os

    os.system("rm -rf build")
    os.system("mkdir build")

    net = spatialSAM()
    data = FLAIR2("train")
    tmp = [v for v in data.paths]
    random.shuffle(tmp)
    tmp = tmp[0:10]
    for i in range(10):
        x, _, y = data.get(tmp[i])
        x = x[[3, 0, 1]]
        torchvision.utils.save_image(x / 255, "build/" + str(i) + "x.png")
        torchvision.utils.save_image(y / 13, "build/" + str(i) + "y.png")

        _, color = net.applySAM(x)
        torchvision.utils.save_image(color / 255, "build/" + str(i) + "z.png")
