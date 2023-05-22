import torch
import os
import segment_anything

import PIL
from PIL import Image, ImageFilter
import numpy

paths = os.listdir("cityscapesample")

with torch.no_grad():
    sam = segment_anything.sam_model_registry["vit_b"](
        checkpoint="model/sam_vit_b_01ec64.pth"
    )
    sam = sam.cuda()

    for path in paths:
        image = PIL.Image.open("cityscapesample/" + path)
        image.save("build/" + path + "_x.png")
        image = numpy.asarray(image.convert("RGB").copy())
        image = numpy.transpose(image, axes=(2, 0, 1))

        x = {}
        x["image"] = torch.Tensor(image).cuda()
        x["original_size"] = (256, 256)
        x["point_coords"] = torch.ones(1, 1, 2).cuda() * 128
        x["point_labels"] = torch.ones(1, 1).cuda()

        out = sam([x], True)[0]
        out = out["masks"][0]
        out, _ = out.max(0)
        out = out.float().cpu().numpy()

        out = PIL.Image.fromarray(numpy.uint8(out != 0) * 255)
        out.save("build/" + path + "_y.png")


patch = PIL.Image.open("patchs/epoch_100_universal_patch.png")
patch = patch.resize((80, 80), PIL.Image.BILINEAR)
patch = patch.filter(PIL.ImageFilter.GaussianBlur(radius=7))
patch = numpy.asarray(patch.convert("RGB").copy())


with torch.no_grad():
    for path in paths:
        image = PIL.Image.open("cityscapesample/" + path)
        image = numpy.asarray(image.convert("RGB").copy())
        image[0:80, 0:80, :] = patch
        tmp = PIL.Image.fromarray(image)
        tmp.save("build/" + path + "_a.png")
        image = numpy.transpose(image, axes=(2, 0, 1))

        x = {}
        x["image"] = torch.Tensor(image).cuda()
        x["original_size"] = (256, 256)
        x["point_coords"] = torch.ones(1, 1, 2).cuda() * 128
        x["point_labels"] = torch.ones(1, 1).cuda()

        out = sam([x], True)[0]
        out = out["masks"][0]
        out, _ = out.max(0)
        out = out.float().cpu().numpy()

        out = PIL.Image.fromarray(numpy.uint8(out != 0) * 255)
        out.save("build/" + path + "_b.png")
