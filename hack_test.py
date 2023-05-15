import torch
import segment_anything
import hack_dataloader

print("load data")
dataset = hack_dataloader.CropExtractor("/scratchf/miniworld/potsdam/test/")

print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.cuda()
    net.eval()


print("test")


def largeforward(net, image, tilesize=256, stride=128):
    pred = torch.zeros(1, 2, image.shape[2], image.shape[3]).cuda()
    image = image.cuda()
    for row in range(0, image.shape[2] - tilesize + 1, stride):
        for col in range(0, image.shape[3] - tilesize + 1, stride):
            tmp = net(image[:, :, row : row + tilesize, col : col + tilesize], False)
            pred[0, :, row : row + tilesize, col : col + tilesize] += tmp[0]
    return pred


cm = torch.zeros((2, 2)).cuda()
with torch.no_grad():
    for i in range(dataset.NB):
        x, y = dataset.getImageAndLabel(i, torchformat=True)
        x, y = x.cuda(), y.cuda().float()

        h, w = y.shape[0], y.shape[1]
        globalresize = torch.nn.AdaptiveAvgPool2d((h, w))
        power2resize = torch.nn.AdaptiveAvgPool2d(((h // 128) * 128, (w // 128) * 128))
        x = power2resize(x)

        z = largeforward(net, x.unsqueeze(0))
        z = globalresize(z)
        z = (z[0, 1, :, :] > z[0, 0, :, :]).float()

        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            cm[a][b] += torch.sum((z == a).float() * (y == b).float())

        if True:
            nextI = len(os.listdir("build"))
            debug = hack_dataloader.torchTOpil(globalresize(x))
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_x.png")
            debug = y * 255
            debug = debug.cpu().numpy()
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_y.png")
            debug = z.cpu().numpy() * 255
            debug = PIL.Image.fromarray(numpy.uint8(debug))
            debug.save("build/" + str(nextI) + "_z.png")

    print("perf=", hack_dataloader.perf(cm))
