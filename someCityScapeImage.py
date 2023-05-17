import torch
import torchvision

if False:
    dataset = torchvision.datasets.Cityscapes("/scratchf/Cityscapes/")
    for i in range(10):
        x, y = dataset[i]
        x.save("build/" + str(i) + "_x.png")
        y.save("build/" + str(i) + "_y.png")


def target_transform(target):
    return torchvision.transforms.functional.to_tensor(
        target[0]
    ), torchvision.transforms.functional.to_tensor(target[1])


# Define the Cityscapes dataset
cityscapes_dataset = torchvision.datasets.Cityscapes(
    root="/scratchf/Cityscapes/",
    split="train",
    mode="fine",
    target_type=["semantic", "instance"],
    transform=torchvision.transforms.ToTensor(),
    target_transform=target_transform,
)

# Create a DataLoader for efficient batched data loading
batch_size = 64
shuffle = False  # Set to True if you want to shuffle the data
num_workers = 4  # Number of subprocesses to use for data loading
data_loader = torch.utils.data.DataLoader(
    cityscapes_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)


def ajust_bounding_box(y, x, r, h, w):
    left = x - r
    top = y - r
    right = x + r
    bottom = y + r

    # Adjust bounding box if it exceeds image boundaries
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, w - 1)
    bottom = min(bottom, h - 1)

    return left, top, right, bottom


# Define a function to extract the square bounding box
def extract_bounding_box(image, sem_labels, ins_labels):
    _, h, w = image.shape

    # Get the pixel coordinates of all pedestrian instances
    pedestrian_indices = torch.nonzero(
        # (sem_labels == 7) &
        (ins_labels > 0),
        as_tuple=False,
    )

    if len(pedestrian_indices) == 0:
        return None  # No pedestrian instances found

    # Select the first pedestrian instance
    pedestrian_idx = pedestrian_indices[0]
    ins_label = ins_labels[pedestrian_idx]

    # Get the bounding box coordinates
    bbox = ins_label.unique(return_inverse=True, sorted=True)[1].reshape(
        ins_label.shape
    )
    ymin, xmin, ymax, xmax = (
        torch.min(bbox[:, 0]),
        torch.min(bbox[:, 1]),
        torch.max(bbox[:, 2]),
        torch.max(bbox[:, 3]),
    )

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Calculate the size of the square bounding box
    r = max(xmax - xmin, ymax - ymin) // 2
    if min(h, w) < 2 * r + 1:
        return None

    left, top, right, bottom = ajust_bounding_box(center_y, center_x, r, h, w)

    return image[top : bottom + 1, left : right + 1]


# Iterate over the data loader to get batches of data
I = 0
for batch in data_loader:
    images, (sem_labels, ins_labels) = batch

    # Iterate over each image in the batch
    for i in range(len(images)):
        image = images[i]

        # Extract the bounding box for the pedestrian
        cropped_image = extract_bounding_box(image, sem_labels[i], ins_labels[i])

        if cropped_image is not None:
            # Resize the cropped image to 256x256 pixels
            resized_image = cropped_image.resize((256, 256))
            torchvision.utils.save_image(resized_image, "build/" + str(I) + ".png")
            I += 1

    quit()
