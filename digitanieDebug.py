import digitanieCommon
import samAsInput
import torch
import torchvision
import skimage


print("load data")
dataset = digitanieCommon.getDIGITANIE("odd")
sam = samAsInput.SAMasInput()


print("test")

good, nbVTs, nbPred = 0, 0, 0
with torch.no_grad():
    for i in range(len(dataset.paths)):
        print(i, "/", len(dataset.paths))
        x, y = dataset.getImageAndLabel(i)
        x, y = x.cuda(), y.cuda()

        vtmap, nbVT = skimage.measure.label(y.cpu().numpy(), return_num=True)
        vtmap = torch.Tensor(vtmap).cuda() - 1

        masks = sam.applySAM(x, True)[0]

        IoU = torch.zeros(masks.shape[0], nbVT)
        for a in range(masks.shape[0]):
            for b in range(nbVT):
                vt = (vtmap == b).float()
                I = masks[a] * vt
                U = masks[a] + vt - I
                IoU[a][b] = I.sum() / (U.sum() + 0.001)

        # assuming a perfect classifier filter the masks
        I = [i for i in range(IoU.shape[0]) if IoU[i].sum() != 0]
        masks = masks[I]
        IoU = IoU[I]

        # still masks could overlap multiple area
        for i in range(masks.shape[0]):
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
        for i in range(IoU.shape[0]):
            if i in I:
                flag[0] += (masks[i]).float()
            else:
                flag[1] += (masks[i]).float()
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

        # return normal function
        good, nbVTs, nbPred = good + len(G), nbVTs + nbVT, nbPred + IoU.shape[0]

        torchvision.utils.save_image(x / 255, "build/" + str(i) + "_x.png")
        torchvision.utils.save_image(y, "build/" + str(i) + "_y.png")
        torchvision.utils.save_image(z, "build/" + str(i) + "_z.png")
        torchvision.utils.save_image(visu / 255, "build/" + str(i) + "_v.png")

    print("sam + perfect classifier perfI=", digitanieCommon.perfI(good, nbVT, nbPred))
