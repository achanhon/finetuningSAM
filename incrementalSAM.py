import torch
import segment_anything
import skimage


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
        x["point_coords"] = torch.Tensor(p).unsqueeze(0).cuda()
        x["point_labels"] = torch.ones(1).cuda()

        with torch.no_grad():
            mask = self.sam([x], False)[0]["masks"][0]

        # take the largest
        mask, _ = mask.max(0)
        mask = (mask > 0).float()

        # erode
        mask = (
            1
            - torch.nn.functional.max_pool2d(
                1 - mask.unsqueeze(0), kernel_size=3, stride=1, padding=1
            )[0]
        )
        return mask

    def sliceMask_(self, x, todo):
        connectedComponent = skimage.measure.label(todo.cpu().numpy())
        blobs = skimage.measure.regionprops(connectedComponent)
        if len(blobs) == 0:
            return None
        tmp = [(region.area, region) for region in blobs]
        tmp = sorted(tmp)
        tmp = tmp[0][1]
        if tmp.area < 10:
            return None

        p = tmp.centroid
        mask = self.samroutine(x, p)
        mask = mask * todo
        if mask.sum() == 0:
            return None
        else:
            return mask

    def sliceMask(self, x, y):
        partition = torch.zeros(y.shape).cuda()
        done = torch.zeros(y.shape).cuda()

        for i in range(1, 100):
            blob = self.sliceMask_(x, y * (1 - done))
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
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 0.0, 255.0],
            [109.0, 98.0, 188.0],
            [230.0, 251.0, 148.0],
            [5.0, 18.0, 230.0],
            [163.0, 38.0, 214.0],
            [173.0, 142.0, 45.0],
            [95.0, 19.0, 120.0],
            [0.0, 0.0, 0.0],
        ]
    ).cuda()

    os.system("rm -r build")
    os.system("mkdir build")
    path = "/d/achanhon/sample_sam_test/"

    image = PIL.Image.open(path + "image.png").convert("RGB").copy()
    mask = PIL.Image.open(path + "mask.png").convert("L").copy()
    image, mask = numpy.asarray(image), numpy.asarray(mask)
    image, mask = torch.Tensor(image), (torch.Tensor(mask) != 255).float()

    torchvision.utils.save_image(image / 255, "build/x.png")
    torchvision.utils.save_image(mask / 255, "build/m.png")
    quit()

    partition = sam.forward(x)

    torchvision.utils.save_image(partition / partition.flatten().max(), "build/p.png")
    torchvision.utils.save_image(getborder(partition), "build/b.png")

    partition = partition.long()
    pseudocolor = torch.zeros(x.shape)
    for row in range(partition.shape[0]):
        for col in range(partition.shape[1]):
            pseudocolor[:, row, col] = palette[partition[row][col] % 9]
    torchvision.utils.save_image(pseudocolor / 255, "build/c.png")
