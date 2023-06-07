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

    def applySAM(self, originaltensor):
        tmp = torch.nn.functional.interpolate(
            input.unsqueeze(0), size=(256), mode="bilinear"
        )
        with torch.no_grad():
            x = {}
            x["image"] = tmp[0].cuda()
            x["original_size"] = (256, 256)
            x["point_coords"] = self.magrille
            x["point_labels"] = self.magrilleL

            masks = sam([x], False)[0]["masks"]
            masks, _ = masks.max(1)
            masks = (masks > 0).float()

            border = self.getborder(masks)
            pseudocolor = self.getpseudocolor(masks)
            return border, pseudocolor

    def getborder(self, masks):
        tmp = masks.unsqueeze(0)
        tmpavg = torch.nn.functional.avg_pool2d(tmp, kernel_size=3, padding=1, stride=1)
        tmp = (tmpavg != 0).float() * (tmpavg != 1).float()
        tmp, _ = tmp[0].max(0)
        return tmp

    def getpseudocolor(masks,flag = False):
        tmp = masks.flatten(1)
        intersectionM = tmp.unsqueeze(0) * tmp.unsqueeze(1)
        unionM = tmp.unsqueeze(0) + tmp.unsqueeze(1) - intersectionM
        intersectionM, unionM = intersectionM.sum(2), unionM.sum(2)
        IoU = intersectionM/(unionM+0.001)
        kept=[]
        for i in range(IoU.shape[0]):
            v,j = max([(IoU[i][j],j) for j in range(i+1,IoU.shape[0])])
            if v>0.8:
                masks[j] = max(masks[j],mask[i])
            else:
                kept.append(i)
        
        if remove==[] or flag:
            
        else:
            return getpseudocolor(masks[kept],True)
         