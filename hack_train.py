import torch
import segment_anything
import dataloader

print("load data")
dataset = dataloader.CropExtractor(sys.argv[1])

print("define model")
net = segment_anything.sam_model_registry["vit_b"](checkpoint="model/sam_vit_b_01ec64.pth"    )
net.hackinit()
net = net.cuda()
net.eval()

print("train")

def diceloss(y, z):
    eps = 0.00001
    z = z.softmax(dim=1)
    z0, z1 = z[:, 0, :, :], z[:, 1, :, :]
    y0, y1 = (y == 0).float(), (y == 1).float()

    inter0, inter1 = (y0 * z0).sum(), (y1 * z1).sum()
    union0, union1 = (y0 + z1 * y0).sum(), (y1 + z0 * y1).sum()
    iou0, iou1 = (inter0 + eps) / (union0 + eps), (inter1 + eps) / (union1 + eps)
    iou = 0.5 * (iou0 + iou1)

    return 1 - iou

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
printloss = torch.zeros(1).cuda()
stats = torch.zeros((2, 2)).cuda()
nbbatchs = 50000
dataset.start()

for i in range(nbbatchs):
    x, y = dataset.getBatch()
    x, y = x.cuda(), y.cuda()
    z = net(x)

    celoss = criterion(z, y)
    dice = diceloss(z, y)
    loss = dice+0.1*celoss

    with torch.no_grad():
        printloss += loss.clone().detach()
        z = (z[:, 1, :, :] > z[:, 0, :, :]).clone().detach().float()
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            stats[a][b] += torch.sum((z == a).float() * (y == b).float())

        if i % 100 == 99:
            print(printloss / 100)
            printloss = torch.zeros(1).cuda()

        if i % 1000 == 999:
            torch.save(net, "build/model.pth")
            perf = dataloader.perf(stats)
            print(i, "perf", perf)
            if perf[0] > 92:
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
