import digitanieCommon
import torch
import os

print("load data")
dataset = digitanieCommon.getDIGITANIE("all")
net = digitanieCommon.EncoderONLY()
net.eval()
net.cuda()

print("train")

CE = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
printloss = torch.zeros(1).cuda()

stats = torch.zeros((2, 2)).cuda()
nbbatchs = 20000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch()
    x, y = x.cuda(), y.cuda()

    z = net(x)

    ce = CE(z, y.long())
    ybis = 1 - torch.nn.functional.max_pool2d(1 - y, kernel_size=3, padding=1, stride=1)
    ce = ce * (1 + 19.0 * (y == 1).float() * (ybis == 0).float())
    loss = ce.mean()

    with torch.no_grad():
        printloss += loss.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        stats += digitanieCommon.confusion(y, z)

        if i < 10:
            print(printloss / (i + 1))

        if i % 100 == 99:
            print(printloss / 100)
            printloss = torch.zeros(1).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            _, iou = digitanieCommon.perf(stats)
            print(i, "perf", iou)
            if iou > 92:
                print("training stops after reaching high training accuracy")
                os._exit(0)
            else:
                stats = torch.zeros((2, 2)).cuda()

    if i > nbbatchs * 0.1:
        loss = loss * 0.5
    if i > nbbatchs * 0.2:
        loss = loss * 0.5
    if i > nbbatchs * 0.5:
        loss = loss * 0.5
    if i > nbbatchs * 0.8:
        loss = loss * 0.5

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
    optimizer.step()

print("training stops after reaching time limit")
os._exit(0)
