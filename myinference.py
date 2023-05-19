import torch
import os
import segment_anything

import PIL
from PIL import Image
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
        print(out["masks"].shape)
        quit()
        out = out["masks"][0][0].float().cpu().numpy()

        out = PIL.Image.fromarray(numpy.uint8(out != 0) * 255)
        out.save("build/" + path + "_y.png")
