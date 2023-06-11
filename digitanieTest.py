import digitanieCommon
import torch
import torchvision
import os

print("load data")
dataset = digitanieCommon.getDIGITANIE("odd")
net = torch.load("build/model.pth")
net.eval()
net.cuda()


print("test")


def largeforward(net, image, tilesize=512, stride=256):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cuda()
    image = image.cuda()
    for row in range(0, image.shape[2] - tilesize + 1, stride):
        for col in range(0, image.shape[3] - tilesize + 1, stride):
            tmp = net(image[:, :, row : row + tilesize, col : col + tilesize])
            pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


cm = torch.zeros((2, 2)).cuda()
good, nbVT, nbPred = 0, 0, 0
with torch.no_grad():
    for i in range(len(dataset.paths)):
        x, y = dataset.getImageAndLabel(i)
        x, y = x.cuda(), y.cuda()

        z = largeforward(net, x.unsqueeze(0))
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
