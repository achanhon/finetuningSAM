import torch
import torchvision

if False:
    dataset = torchvision.datasets.Cityscapes("/scratchf/Cityscapes/",target_type="semantic")
    for i in range(10):
        x, y = dataset[i]
        x.save("build/" + str(i) + "_x.png")
        y.save("build/" + str(i) + "_y.png")
    quit()

def target_transform(target):
    return (torchvision.transforms.functional.to_tensor(target[0]) * 255).long(), (
        torchvision.transforms.functional.to_tensor(target[1])
    ).long()


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


def compute_value_bounding_box(image, k):
    k_indices = torch.where(image == k)

    if len(k_indices[0]) == 0:
        # The value was not found, return None
        return None

    # Compute bounding box coordinates
    min_row = torch.min(k_indices[0])
    min_col = torch.min(k_indices[1])
    max_row = torch.max(k_indices[0])
    max_col = torch.max(k_indices[1])

    return min_row.item(), min_col.item(), max_row.item(), max_col.item()


# Define a function to extract the square bounding box
def extract_bounding_box(image, sem_labels, ins_labels):
    _, h, w = image.shape

    # Get the pixel coordinates of all pedestrian instances
    k = (ins_labels * (sem_labels == 24)).flatten().max()

    if k == 0:
        return None  # No pedestrian instances found

    # Get the bounding box coordinates
    ymin, xmin, ymax, xmax = compute_value_bounding_box(ins_labels, k)

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Calculate the size of the square bounding box
    r = max(xmax - xmin, ymax - ymin) / 2
    if min(h, w) < 2 * r + 1:
        return None

    left, top, right, bottom = ajust_bounding_box(
        int(center_y), int(center_x), int(r), h, w
    )

    return image[:, top : bottom + 1, left : right + 1]


# Iterate over the data loader to get batches of data
I = 0
with torch.no_grad():
    for batch in data_loader:
        images, (sem_labels, ins_labels) = batch

        # Iterate over each image in the batch
        for i in range(len(images)):
            image = images[i]

            # Extract the bounding box for the pedestrian
            cropped_image = extract_bounding_box(image, sem_labels[i], ins_labels[i])

            if cropped_image is not None:
                # Resize the cropped image to 256x256 pixels
                resized_image = torch.nn.functional.interpolate(
                    cropped_image.unsqueeze(0), size=(256, 256), mode="bilinear"
                )[0]
                torchvision.utils.save_image(resized_image, "build/" + str(I) + ".png")
                I += 1

        quit()
