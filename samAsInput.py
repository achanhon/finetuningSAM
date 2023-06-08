import torch
import segment_anything


class SAMasInput:
    def __init__(self):
        self.path = "model/sam_vit_b_01ec64.pth"
        self.sam = segment_anything.sam_model_registry["vit_b"](checkpoint=self.path)
        self.sam = self.sam.cuda()
        self.sam.eval()

        tmp = []
        for row in range(6, 255, 13):
            for col in range(6, 255, 13):
                tmp.append((row, col))
        self.magrille = torch.zeros(len(tmp), 1, 2).cuda()
        self.magrilleL = torch.zeros(len(tmp), 1).cuda()
        for i, (row, col) in enumerate(tmp):
            self.magrille[i][0][0] = row
            self.magrille[i][0][1] = col
            self.magrilleL[i][0] = i

        self.pallette = torch.Tensor(
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

    def applySAM(self, x_):
        tmp = torch.nn.functional.interpolate(
            x_.unsqueeze(0), size=(256), mode="bilinear"
        )
        with torch.no_grad():
            x = {}
            x["image"] = tmp[0].cuda()
            x["original_size"] = (256, 256)
            x["point_coords"] = self.magrille
            x["point_labels"] = self.magrilleL

            masks = self.sam([x], False)[0]["masks"]
            masks, _ = masks.max(1)
            masks = (masks > 0).float()

            border = self.getborder(masks)
            pseudocolor = self.getpseudocolor(masks)

        size_ = (x_.shape[1], x_.shape[2])
        border = torch.nn.functional.interpolate(border.unsqueeze(0), size=size_)
        pseudocolor = torch.nn.functional.interpolate(border.unsqueeze(0), size=size_)
        return border[0], pseudocolor[0]

    def getborder(self, masks):
        tmpavg = torch.nn.functional.avg_pool2d(tmp, kernel_size=3, padding=1, stride=1)
        tmp = (tmpavg != 0).float() * (tmpavg != 1).float()
        tmp, _ = tmp.max(0)
        return tmp

    def getpseudocolor(self, masks, flag=False):
        tmp = masks.flatten(1)
        intersectionM = tmp.unsqueeze(0) * tmp.unsqueeze(1)
        unionM = tmp.unsqueeze(0) + tmp.unsqueeze(1) - intersectionM
        intersectionM, unionM = intersectionM.sum(2), unionM.sum(2)
        IoU = intersectionM / (unionM + 0.001)
        kept = []
        for i in range(IoU.shape[0]):
            v, j = max([(IoU[i][j], j) for j in range(i + 1, IoU.shape[0])])
            if v > 0.8:
                masks[j] = max(masks[j], masks[i])
            else:
                kept.append(i)

        if len(kept) == IoU.shape[0] or flag:
            self.createpseudocolor(masks)
        else:
            return self.getpseudocolor(masks[kept], True)

    def createpseudocolor(self, masks):
        masks = 1 - torch.nn.functional.max_pool2d(
            1 - masks, kernel_size=3, stride=1, padding=1
        )
        out = torch.zeros(256, 256, 3).cuda()
        for i in range(masks.shape[0]):
            out[0] += self.palette[i % 9][0] * masks[i]
            out[1] += self.palette[i % 9][1] * masks[i]
            out[2] += self.palette[i % 9][2] * masks[i]
        return out


if __name__ == "__main__":
    sam = SAMasInput()

    import PIL
    from PIL import Image
    import numpy

    tmp = PIL.Image.open("build/grosse.png")
    tmp = tmp.asarray(tmp.convert("RGB").copy())
    x = torch.Tensor(numpy.transpose(tmp, axes=(2, 0, 1)))
    border, pseudolabel = sam.applySAM(x)

    border = PIL.Image.fromarray(border.cpu().numpy())
    border.save("build/border.png")
    pseudolabel = PIL.Image.fromarray(pseudolabel.cpu().numpy())
    pseudolabel.save("build/pseudolabel.png")
