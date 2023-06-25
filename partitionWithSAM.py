import torch
import segment_anything


def getborder(partition):
    x = partition.unsqueeze(0)
    pMAX = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    x = -x - 1
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    pMIN = -x - 1
    return (pMAX[0] == pMIN[0]).float()


def computeDistance(P, Q):
    # |P|x2 and |Q|x2
    P, Q = P.unsqueeze(1), Q.unsqueeze(0)  # 1x|P|x2 and |Q|x1x2
    D = (P - Q) ** 2  # |P|x|Q|x2
    return D.sum(2)  # |P|x|Q|


def nearestCloud(P, Q, I):
    D = computeDistance(P, Q)
    _, D = D.min(1)
    return I[D]


class PartitionSAM:
    def __init__(self):
        self.path = "model/sam_vit_b_01ec64.pth"
        self.sam = segment_anything.sam_model_registry["vit_b"](checkpoint=self.path)
        self.sam = self.sam.cuda()
        self.sam.eval()

        tmp = []
        for row in range(8, 248, 17):
            for col in range(8, 248, 17):
                tmp.append((row, col))
        print("SAM GRID SIZE", len(tmp))

        self.magrille = torch.zeros(len(tmp), 1, 2).cuda()
        self.magrilleL = torch.zeros(len(tmp), 1).cuda()
        for i, (row, col) in enumerate(tmp):
            self.magrille[i][0][0] = row
            self.magrille[i][0][1] = col
            self.magrilleL[i][0] = i
        self.magrille = self.magrille.long()
        self.magrilleL = self.magrilleL.long()

    def rawSAM(self, x_):
        assert x_.shape == (3, 256, 256)
        x = {}
        x["image"] = x_.cuda()
        x["original_size"] = (256, 256)
        x["point_coords"] = self.magrille
        x["point_labels"] = self.magrilleL

        with torch.no_grad():
            masks = self.sam([x], False)[0]["masks"]

        # take the largest
        masks, _ = masks.max(1)
        masks = (masks > 0).float()

        # erode
        masks = 1 - torch.nn.functional.max_pool2d(
            1 - masks, kernel_size=3, stride=1, padding=1
        )
        return masks

    def intersectMaskCenters(self, mask, centers):
        m = mask[centers[:, 0], centers[:, 1]].sum()
        if centers.shape[0] <= 3:
            return m >= 2
        else:
            return m >= 0.7 * centers.shape[1]

    def mergingMasks_(self, masks, centers):
        for i in range(masks.shape[0]):
            for j in range(i + 1, masks.shape[0]):
                a = self.intersectMaskCenters(masks[i], centers[j])
                b = self.intersectMaskCenters(masks[j], centers[i])
                if a and b:
                    centers[i] = torch.cat([centers[i], centers[j]], dim=0)
                    del centers[j]
                    masks[i] = torch.clamp(masks[i] + masks[j], 0, 1)
                    I = [k for k in range(masks.shape[0]) if k != j]
                    masks = masks[I]
                    return self.mergingMasks_(masks, centers)

        return masks, centers

    def mergingMasks256(self, masks):
        assert masks.shape[0] == self.magrille.shape[0]
        tmp = [self.magrille[i].clone() for i in range(masks.shape[0])]
        return self.mergingMasks_(masks, tmp)

    def applySAM(self, x_):
        _, h, w = x_.shape
        x = torch.nn.functional.interpolate(
            x_.unsqueeze(0), size=(256, 256), mode="bilinear"
        )
        masks = self.rawSAM(x[0])
        masks, centers = self.mergingMasks256(masks)

        issues, ok = (masks.sum(0) != 1).float(), (masks.sum(0) == 1).float()
        print(issues.sum(), ok.sum())

        for i in range(masks.shape[0]):
            masks[i] *= i
        partition = ok * masks.sum(0)

        P = torch.nonzero(issues)
        I = []
        for i in range(masks.shape[0]):
            I.append(torch.ones(centers[i].shape[0]) * i)
        I = torch.cat(I, dim=0).cuda()
        centers = torch.cat(centers, dim=0)
        J = nearestCloud(P, centers, I)
        partition[P[:, 0], P[:, 1]] = J

        partition = partition.unsqueeze(0).unsqueeze(0)
        partition = torch.nn.functional.interpolate(
            partition, size=(h, w), mode="nearest"
        )
        return partition[0][0]

    def forward(self, x):
        with torch.no_grad():
            return self.applySAM(x)


if __name__ == "__main__":
    sam = PartitionSAM()

    import numpy
    import rasterio
    import os
    import torchvision

    palette = torch.Tensor(
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

    os.system("rm -r build")
    os.system("mkdir build")
    path = "/scratchf/AI4GEO/DIGITANIE/Arcachon/Arcachon_EPSG32630_4.tif"

    with rasterio.open(path) as src:
        r = numpy.clip(src.read(1) * 2, 0, 1)
        g = numpy.clip(src.read(2) * 2, 0, 1)
        b = numpy.clip(src.read(3) * 2, 0, 1)
        x = numpy.stack([r, g, b], axis=0) * 255
        x = torch.Tensor(x).cuda()

    torchvision.utils.save_image(x / 255, "build/x.png")

    partition = sam.forward(x)

    torchvision.utils.save_image(partition / partition.flatten().max(), "build/p.png")
    torchvision.utils.save_image(getborder(partition), "build/b.png")

    partition = partition.long()
    pseudocolor = torch.zeros(x.shape)
    for row in range(partition.shape[0]):
        for col in range(partition.shape[1]):
            pseudocolor[:, row, col] = palette[partition[row][col] % 9]
    torchvision.utils.save_image(pseudocolor / 255, "build/c.png")
