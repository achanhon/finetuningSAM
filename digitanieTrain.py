import digitanieCommon
import torch
import os

print("load data")
dataset = digitanieCommon.getDIGITANIE("even")
# net = digitanieCommon.Deeplab()
# perf= (tensor(92.3988, device='cuda:0'), tensor(75.0972, device='cuda:0'))
# perfI= (0.413362199658613, 0.7784599551437283, 0.530999953083384)
# on retrouve enfin un truc normal
net = digitanieCommon.FUSION()
print("deeplab")
net.eval()
net.cuda()

print("train")

CE = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
printloss = torch.zeros(1).cuda()

stats = torch.zeros((2, 2)).cuda()
nbbatchs = 30000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch()
    x, y = x.cuda(), y.cuda()

    z = net(x)

    ce = CE(z, y.long())
    ybis = torch.nn.functional.max_pool2d(y, kernel_size=3, padding=1, stride=1)
    ce = ce * (1 + 19.0 * (y == 0).float() * (ybis == 1).float())
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
