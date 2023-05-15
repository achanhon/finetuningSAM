import torch
import torchvision
import segment_anything

with torch.no_grad():
    #sam = segment_anything.sam_model_registry["vit_h"](checkpoint="build/sam_vit_h_4b8939.pth").cuda()
    sam = segment_anything.sam_model_registry["vit_b"](checkpoint="build/sam_vit_b_01ec64.pth").cuda()
    sam.hackinit()
    x = {}
    x["image"] = torch.rand(3,256,256).cuda()
    x["original_size"]=(256,256)
    xx = {}
    xx["image"] = torch.rand(3,256,256).cuda()
    xx["original_size"]=(256,256)
    print(sam([x,xx],False).shape)