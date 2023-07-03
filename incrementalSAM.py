import torch
import segment_anything
import skimage


def centerOfMask(y_):
    y = (y_ != 1).unsqueeze(0).float()
    assert (y == 0).float().sum() > 0
    ybefore = y.clone()
    for i in range(300):
        y = torch.nn.functional.max_pool2d(y, kernel_size=3, padding=1, stride=1)
        if (y == 0).float().sum() == 0:
            break
        else:
            ybefore = y.clone()
    return torch.nonzero((ybefore[0] == 0).float())[0]


class IncrementalSAM:
    def __init__(self):
        self.path = "model/sam_vit_b_01ec64.pth"
        self.sam = segment_anything.sam_model_registry["vit_b"](checkpoint=self.path)
        self.sam = self.sam.cuda()
        self.sam.eval()

    def samroutine(self, x_, p):
        assert x_.shape == (3, 256, 256)
        x = {}
        x["image"] = x_.cuda()
        x["original_size"] = (256, 256)
        x["point_coords"] = torch.Tensor(p).unsqueeze(0).unsqueeze(0).cuda()
        x["point_labels"] = torch.ones(1, 1).cuda()

        with torch.no_grad():
            mask = self.sam([x], False)[0]["masks"][0]

        # take the largest
        mask, _ = mask.max(0)
        mask = (mask > 0).float()

        # erode
        mask = 1 - torch.nn.functional.max_pool2d(
            1 - mask.unsqueeze(0), kernel_size=3, stride=1, padding=1
        )
        return mask[0]

    def sliceMask_(self, x, todo):
        if False:
            connectedComponent = skimage.measure.label(todo.cpu().numpy())
            blobs = skimage.measure.regionprops(connectedComponent)
            if len(blobs) == 0:
                return None
            tmp = [(-blobs[i].area, i) for i in range(len(blobs))]
            tmp = sorted(tmp)
            tmp = blobs[tmp[0][1]]
            print("blob size", tmp.area)
            if tmp.area < 10:
                return None
            p = tmp.centroid
        p = centerOfMask(todo)
        print(p)
        mask = self.samroutine(x, [p[1], p[0]])
        print("mask size", mask.sum())
        mask = mask * todo
        print("mask size after removing bad part", mask.sum())
        if mask.sum() < 10:
            return None
        else:
            return mask

    def sliceMask(self, x, y):
        with torch.no_grad():
            partition = torch.zeros(y.shape).cuda()
            done = torch.zeros(y.shape).cuda()

            for i in range(1, 100):
                blob = self.sliceMask_(x * (1 - done), y * (1 - done))
                if blob is None:
                    return partition
                partition = partition + blob * i
                done = done + (blob > 0).float()
            return partition


if __name__ == "__main__":
    sam = IncrementalSAM()

    import numpy
    import rasterio
    import PIL
    from PIL import Image
    import os
    import torchvision

    palette = torch.Tensor(
        [
            [0.0, 0.0, 0.0],
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 0.0, 255.0],
            [109.0, 98.0, 188.0],
            [230.0, 251.0, 148.0],
            [5.0, 18.0, 230.0],
            [163.0, 38.0, 214.0],
            [173.0, 142.0, 45.0],
            [95.0, 19.0, 120.0],
        ]
    ).cuda()

    os.system("rm -r build")
    os.system("mkdir build")
    path = "/d/achanhon/testAI4GEOsam/"
    x = "NEW-YORK_20180524_T_TOA_reproj-EPSG:32618_cog.tif"
    m = "NEW-YORK_20180524_T_8192_7680_512_256_nCOS1_building_labels.tif"

    with rasterio.open(path + x) as src:
        transform = src.transform
        x, y = (584385.4, 4508194.2)
        pixel_x = int((x - transform[2]) / transform[0])
        pixel_y = int((y - transform[5]) / transform[4])

        r = numpy.clip(src.read(1) * 2, 0, 1)
        print(r.shape, pixel_x, pixel_y)

        quit()
        g = numpy.clip(src.read(2), 0, 1)
        b = numpy.clip(src.read(3) * 2, 0, 1)
        x = numpy.stack([r, g, b], axis=0) * 255

    quit()
    path = "/d/achanhon/sample_sam_test/"

    image = PIL.Image.open(path + "image.png").convert("RGB").copy()
    image = numpy.transpose(numpy.asarray(image.resize((256, 256))), axes=(2, 0, 1))
    mask = PIL.Image.open(path + "mask.png").convert("L").copy()
    mask = numpy.asarray(mask.resize((256, 256)))
    image, mask = torch.Tensor(image), (torch.Tensor(mask) != 255).float()

    torchvision.utils.save_image(image / 255, "build/x.png")
    torchvision.utils.save_image(mask, "build/m.png")

    partition = sam.sliceMask(image.cuda(), mask.cuda())

    torchvision.utils.save_image(partition / partition.flatten().max(), "build/p.png")

    partition = partition.long()
    pseudocolor = torch.zeros(image.shape)
    for row in range(partition.shape[0]):
        for col in range(partition.shape[1]):
            pseudocolor[:, row, col] = palette[partition[row][col] % 9]
    torchvision.utils.save_image(pseudocolor / 255, "build/c.png")
