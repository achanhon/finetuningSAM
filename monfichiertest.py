import torch
import torchvision
import os

if "build" not in os.listdir("."):
    os.system("mkdir build")
    os.system("cd build ; wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

import segment_anything

with torch.no_grad():
    sam = segment_anything.sam_model_registry["vit_h"](checkpoint="build/sam_vit_h_4b8939.pth").cuda()
    x = {}
    x["image"] = torch.rand(3,256,256).cuda()
    x["original_size"]=(256,256)
    xx = {}
    xx["image"] = torch.rand(3,256,256).cuda()
    xx["original_size"]=(256,256)
    print(sam([x,xx],False))