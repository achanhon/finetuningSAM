import torch
import torchvision

if False:
    dataset = torchvision.datasets.Cityscapes(
        "/scratchf/Cityscapes/", target_type="semantic"
    )
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
    # Calculate the minimum and maximum coordinates for the crop box
    min_y = max(y - r, 0)
    max_y = min(y + r, h - 1)
    min_x = max(x - r, 0)
    max_x = min(x + r, w - 1)

    # Calculate the dimensions of the crop box
    crop_h = max_y - min_y + 1
    crop_w = max_x - min_x + 1

    # Adjust the crop box to ensure it remains 2r x 2r
    crop_h_diff = max(0, crop_h - 2 * r)
    crop_w_diff = max(0, crop_w - 2 * r)
    min_y += crop_h_diff // 2
    max_y -= crop_h_diff // 2
    min_x += crop_w_diff // 2
    max_x -= crop_w_diff // 2

    # Calculate the coordinates of the top-left corner of the crop box
    crop_y = y - min_y - r
    crop_x = x - min_x - r

    return crop_y, crop_x, 2 * r, 2 * r


def compute_value_bounding_box(image, k):
    k_indices = torch.where(image == k)

    # Compute bounding box coordinates
    min_row = torch.min(k_indices[0])
    min_col = torch.min(k_indices[1])
    max_row = torch.max(k_indices[0])
    max_col = torch.max(k_indices[1])

    return int(min_row), int(min_col), int(max_row), int(max_col)


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
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Calculate the size of the square bounding box
    r = max(xmax - xmin, ymax - ymin) // 2
    if min(h, w) < 2 * r + 1:
        return None

    left, top, _, _ = ajust_bounding_box(center_y, center_x, r, h, w)

    return image[:, top : top + 2 * r + 1, left : left + 2 * r + 1]


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
