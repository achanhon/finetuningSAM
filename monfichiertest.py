import torch
import torchvision
import os

if "build" not in os.listdir("."):
    os.system("mkdir build")
    os.system("cd build ; wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

import segment_anything

sam = segment_anything.sam_model_registry["vit_h"](checkpoint="build/sam_vit_h_4b8939.pth")
print(sam)