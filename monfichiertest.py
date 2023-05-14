import torch
import torchvision

print("ok")

import sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
print(sam)