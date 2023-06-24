import torch
import segment_anything


def computeDistance(P,Q):
    # |P|x2 and |Q|x2
    P,Q = P.unsqueeze(0),Q.unsqueeze(1) #1x|P|x2 and |Q|x1x2
    D = (P - Q) ** 2  # |P|x|Q|x2
    return D.sum(2)  # |P|x|Q|

def computeDistanceToCloud(P,Q,I):
    DC = torch.zeros(P.shape[0],len(Qs))
    for i,Q in enumerate(Qs):
        D = computeDistance(P,Q)
        D,_ = D.min(1)
        DC[:,i] = D
    _, I = DC.min(1)
    return I 


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

        self.allpixels = []
        for row in range(256):
            for col in range(256):
                tmp = torch.zeros(1,2)
                tmp[0][0]=row
                tmp[0][1]=col
                self.allpixels.append(tmp)
        self.allpixels = torch.cat(self.allpixels,dim=0)

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

    def rawSAM(self,x):
        assert x.shape == (3, 256, 256)
        x = {}
        x["image"] = x_
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
        return mask[centers[:, 0], centers[:, 1]].sum() >= 1

    def mergingMasks_(self, masks, centers):
        for i in range(masks.shape[0]):
            for j in range(i + 1, masks.shape[0]):
                a = self.intersectMaskCenters(masks[i], centers[j])
                b = self.intersectMaskCenters(                    masks[j],centers[i]              )
                if a and b:
                    centers[i] = torch.cat(centers[i],centers[j].clone(),dim=0)
                    del centers[j]
                    masks[i] = torch.clamp(masks[i] + masks[j], 0, 1)
                    del masks[j]
                    return self.mergingMasks_(masks, centers)

        return masks, centers

    def mergingMasks(self, masks):
        assert masks.shape[0] == self.magrille.shape[0]
        tmp = [self.magrille[i].unsqueeze(0) for i in range(masks.shape[0])]
        return self.mergingMasks_(masks, tmp)
    
    def applySAM256(self,x):
        if len(x.shape==3):
            masks = self.rawSAM(x)
            masks = self.mergingMasks(masks)

            self.all

        else:
            out = []
            for i in range(x.shape[0])
    

    def applySAMmultiple(self, x):
        b, _, h, w = x.shape
        W, H = (w // 256) * 256 + 256, (h // 256) * 256 + 256

        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear")

        masks = torch.zeros(b, 1, w, h).cuda()
        nbM = torch.zeros(b).cuda()
        for i in range(b):
            nbM[i], masks[i] = self.applySAM(x[i])
        masks = torch.nn.functional.interpolate(masks, size=(h, w), mode="nearest")

        colorM = torch.zeros(b, 3, w, h).cuda()
        for i in range(b):
            for j in range(nbM[i]):
                colorM[i] += self.palette[j] * (masks[i] == j).float()
        return nbM, masks, colorM

    def applySAM(self, x_):
        _, H, W = x_.shape
        largeMasks, largegrid, largeMerged = [], [], []
        for w in range(0, W - 256, 256):
            for h in range(0, H - 256, 256):
                masks, grid, merged = self.basicSAM(x[:, h : h + 256, w : w + 256])
                tmp = torch.zeros(masks.shape[0], H, W).cuda()
                tmp[:, h : h + 256, w : w + 256] = masks
                largeMasks.append(tmp.clone())
                tmp = grid.clone()
                tmp[:, 0] += h
                tmp[:, 1] += w
                largegrid.append(grid)

        # NMS
        tmp = [(masks[i].sum(), i) for i in range(masks.shape[0])]
        tmp = sorted(tmp)
        remove = []
        for i in range(len(tmp)):
            if tmp[i][0] == 0:
                remove.append(i)
                break
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

        size_ = (x_.shape[1], x_.shape[2])
        if masks.shape[0] == 0:
            if debug:
                return torch.zeros(x_.shape).cuda()
            else:
                return torch.zeros(size_).cuda(), torch.zeros(x_.shape).cuda()

        # border and pseudo color
        border = self.getborder(masks).unsqueeze(0).unsqueeze(0)
        pseudocolor = self.getpseudocolor(masks).unsqueeze(0)

        border = torch.nn.functional.interpolate(border, size=size_, mode="bilinear")
        pseudocolor = torch.nn.functional.interpolate(pseudocolor, size=size_)
        if debug:
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=size_)
            return masks[0]
        else:
            return border[0][0], pseudocolor[0]

    def getborder(self, masks):
        tmp = torch.nn.functional.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        tmp = (tmp == 1).float() * (masks == 0).float()
        tmp, _ = tmp.max(0)
        return tmp

    def getpseudocolor(self, masks):
        out = torch.zeros(3, 256, 256).cuda()
        for i in range(masks.shape[0]):
            tmp = torch.zeros(3, 256, 256).cuda()
            tmp[0] = self.palette[i % 9][0] * masks[i]
            tmp[1] = self.palette[i % 9][1] * masks[i]
            tmp[2] = self.palette[i % 9][2] * masks[i]
            out = torch.max(out, tmp)
        return out


if __name__ == "__main__":
    sam = SAMasInput()

    import PIL
    from PIL import Image
    import numpy
    import rasterio
    import os

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
