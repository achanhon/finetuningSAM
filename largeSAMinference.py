import torch
import segment_anything


def computeDistance(P, Q):
    # |P|x2 and |Q|x2
    P, Q = P.unsqueeze(0), Q.unsqueeze(1)  # 1x|P|x2 and |Q|x1x2
    D = (P - Q) ** 2  # |P|x|Q|x2
    return D.sum(2)  # |P|x|Q|


def nearestCloud(P, Q, I):
    P, Q = P.unsqueeze(0), Q.unsqueeze(1)
    D = (P - Q) ** 2
    _, D = D.min(1)
    return I[D]


class SAMwithoutResizing:
    def __init__(self):
        self.path = "model/sam_vit_b_01ec64.pth"
        self.sam = segment_anything.sam_model_registry["vit_b"](checkpoint=self.path)
        self.sam = self.sam.cuda()
        self.sam.eval()

        tmp = []
        for row in range(8, 255, 17):
            for col in range(8, 255, 17):
                tmp.append((row, col))
        self.magrille = torch.zeros(len(tmp), 1, 2).cuda()
        self.magrilleL = torch.zeros(len(tmp), 1).cuda()
        for i, (row, col) in enumerate(tmp):
            self.magrille[i][0][0] = row
            self.magrille[i][0][1] = col
            self.magrilleL[i][0] = i

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
        if centers.shape[0] <= 2:
            return m >= 1
        else:
            return m >= 0.6 * centers.shape[1]

    def mergingMasks_(self, masks, centers):
        for i in range(masks.shape[0]):
            for j in range(i + 1, masks.shape[0]):
                a = self.intersectMaskCenters(masks[i], centers[j])
                b = self.intersectMaskCenters(masks[j], centers[i])
                if a and b:
                    centers[i] = torch.cat(centers[i], centers[j].clone(), dim=0)
                    del centers[j]
                    masks[i] = torch.clamp(masks[i] + masks[j], 0, 1)
                    del masks[j]
                    return self.mergingMasks_(masks, centers)

        return masks, centers

    def mergingMasks256(self, masks):
        assert masks.shape[0] == self.magrille.shape[0]
        tmp = [self.magrille[i].unsqueeze(0) for i in range(masks.shape[0])]
        return self.mergingMasks_(masks, tmp)

    def applySAM256(self, x):
        assert x.shape == (3, 256, 256)
        masks = self.rawSAM(x)
        masks, centers = self.mergingMasks256(masks)

        issues = (masks.sum(0) != 1).float()

        P = torch.nonzero(issues)
        I = []
        for i in range(masks.shape[0]):
            I.append(torch.ones(centers[i].shape[0]) * i)
        I = torch.cat(I, dim=0)
        centers = torch.cat(centers, dim=0)
        J = nearestCloud(P, centers, I)

        partition = torch.zeros(256, 256)
        partition[P[:, 0], P[:, 1]] = J
        masks = masks * torch.arange(masks.shape[0]).unsqueeze(-1).unsqueeze(-1)
        partition += masks.sum(0) * (issues == 0).float()
        return partition

    def applySAM256multiple(self, x):
        partition = [self.applySAM256(x[i]) for i in range(x.shape[0])]
        return torch.stack(partition, dim=0)

    def applySAM(self, x):
        _, h, w = x.shape
        W, H = (w // 192) * 192 + 256, (h // 192) * 192 + 256

        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear")

        largeMasks, largegrid = [], []
        for w in range(0, W - 256, 192):
            for h in range(0, H - 256, 192):
                masks = self.rawSAM(x[:, h : h + 256, w : w + 256])
                masks, centers = self.mergingMasks256(masks)
                for i in range(masks.shape[0]):
                    centers[i][0] += h
                    centers[i][1] += w
                tmp = torch.zeros(masks.shape[0], H, W).cuda()
                tmp[:, h : h + 256, w : w + 256] = masks
                largegrid.extends(centers)
                largeMasks.appedn(tmp)
        largeMasks = torch.cat(largeMasks, dim=0)

        masks, centers = self.mergingMasks_(masks, largegrid)

        issues = (masks.sum(0) != 1).float()

        P = torch.nonzero(issues)
        I = []
        for i in range(masks.shape[0]):
            I.append(torch.ones(centers[i].shape[0]) * i)
        I = torch.cat(I, dim=0)
        centers = torch.cat(centers, dim=0)
        J = nearestCloud(P, centers, I)

        partition = torch.zeros(H, W).cuda()
        partition[P[:, 0], P[:, 1]] = J
        masks = masks * torch.arange(masks.shape[0]).unsqueeze(-1).unsqueeze(-1)
        partition += masks.sum(0) * (issues == 0).float()

        _, h, w = x.shape
        return torch.nn.functional.interpolate(partition, size=(h, w), mode="bilinear")

    def applySAMmultiple(self, x):
        partition = [self.applySAM(x[i]) for i in range(x.shape[0])]
        return torch.stack(partition, dim=0)

    def getborder(self, masks):
        tmp = torch.nn.functional.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        tmp = (tmp == 1).float() * (masks == 0).float()
        tmp, _ = tmp.max(0)
        return tmp


if __name__ == "__main__":
    sam = SAMwithoutResizing()

    import PIL
    from PIL import Image
    import numpy
    import rasterio
    import os

    print("TODO")

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

    with rasterio.open(
        "/scratchf/AI4GEO/DIGITANIE/Arcachon/Arcachon_EPSG32630_4.tif"
    ) as src:
        r = numpy.clip(src.read(1) * 2, 0, 1)
        g = numpy.clip(src.read(2) * 2, 0, 1)
        b = numpy.clip(src.read(3) * 2, 0, 1)
        x = numpy.stack([r, g, b], axis=0) * 255
        tmp = numpy.stack([r, g, b], axis=-1) * 255

    tmp = PIL.Image.fromarray(numpy.uint8(tmp))
    tmp.save("build/origine.png")
    x = torch.Tensor(x)
    with torch.no_grad():
        border, pseudolabel = sam.applySAM(x)

    tmp = PIL.Image.fromarray(numpy.uint8(border.cpu().numpy() * 255))
    tmp.save("build/border.png")
    tmp = numpy.transpose(pseudolabel.cpu().numpy(), axes=(1, 2, 0))
    tmp = PIL.Image.fromarray(numpy.uint8(tmp))
    tmp.save("build/pseudolabel.png")
