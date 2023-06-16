import digitanieCommon
import samAsInput
import torch
import torchvision
import os

print("load data")
dataset = digitanieCommon.getDIGITANIE("odd")
sam = samAsInput.SAMasInput()


print("test")

good, nbVT, nbPred = 0, 0, 0
with torch.no_grad():
    for i in range(len(dataset.paths)):
        print(i, "/", len(dataset.paths))
        x, y = dataset.getImageAndLabel(i)
        x, y = x.cuda(), y.cuda()

        masks = sam.applySAM(x,True)
        z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

        cm += digitanieCommon.confusion(y, z)
        g, vt, pred, visu = digitanieCommon.confusionInstance(y, z)
        good, nbVT, nbPred = good + g, nbVT + vt, nbPred + pred

        if True:
            torchvision.utils.save_image(x / 255, "build/" + str(i) + "_x.png")
            torchvision.utils.save_image(y, "build/" + str(i) + "_y.png")
            torchvision.utils.save_image(z, "build/" + str(i) + "_z.png")
            torchvision.utils.save_image(visu / 255, "build/" + str(i) + "_v.png")

    print("perf=", digitanieCommon.perf(cm))
    print("perfI=", digitanieCommon.perfI(good, nbVT, nbPred))
