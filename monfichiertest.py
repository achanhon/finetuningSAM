import torch
import segment_anything

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


print("version originale")
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

    print([label for label in sam([x, xx])[0]])

print("version originale avec une vrai image")
import PIL
from PIL import Image
import numpy

with torch.no_grad():
    tmp = PIL.Image.open("/scratchf/miniworld/potsdam/train/0_x.png")
    tmp.save("build/x.png")

    tmp = numpy.asarray(tmp.convert("RGB").copy())
    h, w, c = tmp.shape
    tmp = tmp[h // 2 - 128 : h // 2 + 128, w // 2 - 128 : w // 2 + 128, :]
    tmp = torch.Tensor(numpy.transpose(tmp, axes=(2, 0, 1)))

    x = {}
    x["image"] = tmp.cuda()
    x["original_size"] = (256, 256)
    x["point_coords"] = torch.ones(1,1,2).cuda()*110
    x["point_labels"] = torch.ones(1,1).cuda()


    out = sam([x],)[0]
    out = out["masks"][0][0].float().cpu().numpy()
    
    tmp = PIL.Image.fromarray(numpy.uint8(out!=0)*255)
    tmp.save("build/y.png")
