import torch
import segment_anything


class SAMasInput:
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

    def applySAMmultiple(self, x):
        with torch.no_grad():
            xc = torch.ones(x.shape[0], 3, x.shape[2], x.shape[3]).cuda()
            xb = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).cuda()
            for i in range(x.shape[0]):
                b, c = self.applySAM(x[i])
                xc[i] = c
                xb[i] = b
        return torch.cat([x, xc], dim=1), xb

    def applySAM(self, x_, debug=False):
        tmp = torch.nn.functional.interpolate(
            x_.unsqueeze(0), size=(256, 256), mode="bilinear"
        )
        x = {}
        x["image"] = tmp[0].cuda()
        x["original_size"] = (256, 256)
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
