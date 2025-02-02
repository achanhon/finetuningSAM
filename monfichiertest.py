import torch
import segment_anything

if False:
    print("version tenseur")
    with torch.no_grad():
        sam = segment_anything.sam_model_registry["vit_b"](
            checkpoint="model/sam_vit_b_01ec64.pth"
        )
        sam.hackinit()
        sam = sam.cuda()
        sam.eval()

        x = torch.rand(2, 3, 256, 256).cuda()
        print(sam(x).shape)


print("version originale 512")
with torch.no_grad():
    sam = segment_anything.sam_model_registry["vit_b"](
        checkpoint="model/sam_vit_b_01ec64.pth"
    )
    sam = sam.cuda()

    x = {}
    x["image"] = torch.rand(3, 512, 512).cuda()
    x["original_size"] = (512, 512)
    xx = {}
    xx["image"] = torch.rand(3, 512, 512).cuda()
    xx["original_size"] = (512, 512)

    print(sam([x, xx])[0]["masks"].shape)


print("version originale 256")
with torch.no_grad():
    sam = segment_anything.sam_model_registry["vit_b"](
        checkpoint="model/sam_vit_b_01ec64.pth"
    )
    sam = sam.cuda()

    x = {}
    x["image"] = torch.rand(3, 256, 256).cuda()
    x["original_size"] = (256, 256)
    xx = {}
    xx["image"] = torch.rand(3, 256, 256).cuda()
    xx["original_size"] = (256, 256)

    print(sam([x, xx])[0]["masks"].shape)

print("version originale avec une vrai image")
import PIL
from PIL import Image
import numpy

with torch.no_grad():
    tmp = PIL.Image.open("/scratchf/miniworld/potsdam/train/0_x.png")

    tmp = numpy.asarray(tmp.convert("RGB").copy())
    h, w, c = tmp.shape
    tmp = tmp[h // 2 - 128 : h // 2 + 128, w // 2 - 128 : w // 2 + 128, :]
    tmpbis = PIL.Image.fromarray(tmp)
    tmpbis.save("build/x.png")
    tmp = torch.Tensor(numpy.transpose(tmp, axes=(2, 0, 1)))

    x = {}
    x["image"] = tmp.cuda()
    x["original_size"] = (256, 256)
    x["point_coords"] = torch.ones(1, 1, 2).cuda() * 110
    x["point_labels"] = torch.ones(1, 1).cuda()

    out = sam([x], False)[0]
    out = out["masks"][0][0].float().cpu().numpy()

    tmp = PIL.Image.fromarray(numpy.uint8(out != 0) * 255)
    tmp.save("build/y.png")

print("version originale avec une vrai image et plusieurs points")
tmp = []
for row in range(16, 255, 32):
    for col in range(16, 255, 32):
        tmp.append((row, col))
magrille = torch.zeros(len(tmp), 1, 2).cuda()
magrilleL = torch.zeros(len(tmp), 1).cuda()
for i, (row, col) in enumerate(tmp):
    magrille[i][0][0] = row
    magrille[i][0][1] = col
    magrilleL[i][0] = i

with torch.no_grad():
    tmp = PIL.Image.open("/scratchf/miniworld/potsdam/train/0_x.png")

    tmp = numpy.asarray(tmp.convert("RGB").copy())
    h, w, c = tmp.shape
    tmp = tmp[h // 2 - 128 : h // 2 + 128, w // 2 - 128 : w // 2 + 128, :]
    tmpbis = PIL.Image.fromarray(tmp)
    tmpbis.save("build/x.png")
    tmp = torch.Tensor(numpy.transpose(tmp, axes=(2, 0, 1)))

    x = {}
    x["image"] = tmp.cuda()
    x["original_size"] = (256, 256)
    x["point_coords"] = magrille
    x["point_labels"] = magrilleL

    out = sam([x], False)[0]["masks"]
    out, _ = out.max(1)
    out = out.cpu().numpy()

    for i in range(magrille.shape[0]):
        tmp = PIL.Image.fromarray(numpy.uint8(out[i] != 0) * 255)
        tmp.save("build/y" + str(i) + ".png")

import os

os.system("rm build/x*")
os.system("rm build/y*")

print("wtf")
tmp = []
for row in range(6, 255, 12):
    for col in range(6, 255, 12):
        tmp.append((row, col))
magrille = torch.zeros(len(tmp), 1, 2).cuda()
magrilleL = torch.zeros(len(tmp), 1).cuda()
for i, (row, col) in enumerate(tmp):
    magrille[i][0][0] = row
    magrille[i][0][1] = col
    magrilleL[i][0] = i
print(i)

with torch.no_grad():
    tmp = PIL.Image.open("build/grosse.png")
    tmpIMAGE = tmp.resize((256, 256))

    tmp = numpy.asarray(tmpIMAGE.convert("RGB").copy())
    tmp = torch.Tensor(numpy.transpose(tmp, axes=(2, 0, 1)))

    x = {}
    x["image"] = tmp.cuda()
    x["original_size"] = (256, 256)
    x["point_coords"] = magrille
    x["point_labels"] = magrilleL

    out = sam([x], False)[0]["masks"]
    out, _ = out.max(1)
    out = out.cpu().numpy()

    for i in range(magrille.shape[0]):
        tmp = PIL.Image.fromarray(numpy.uint8(out[i] != 0) * 255)
        tmpIMAGE.save("build/" + str(i) + "x.png")
        tmp.save("build/" + str(i) + "y.png")
