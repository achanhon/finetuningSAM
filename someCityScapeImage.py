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
    tmp = ins_labels * (sem_labels == 24)
    k = tmp.flatten().max()

    if k == 0:
        return None  # No pedestrian instances found
    else:
        l = sorted([(-(tmp == (r + 1)).sum(), r + 1) for r in range(k)])
        k = l[0][1]

    # Get the bounding box coordinates
    ymin, xmin, ymax, xmax = compute_value_bounding_box(ins_labels, k)

    # Calculate the center of the bounding box
    x = (xmin + xmax) // 2
    y = (ymin + ymax) // 2
    r = max(xmax - xmin, ymax - ymin) // 2
    if r < 128:
        return None  # No pedestrian enough large
    r = min(int(r * 1.2), 230)

    y = min(max(r + 1, y), h - r - 2)
    x = min(max(r + 1, x), w - r - 2)

    out1 = image[:, y - r : y + r + 1, x - r : x + r + 1]
    out2 = (ins_labels[y - r : y + r + 1, x - r : x + r + 1] == k).float()
    return out1, out2


# Iterate over the data loader to get batches of data
I = 0
with torch.no_grad():
    for batch in data_loader:
        images, (sem_labels, ins_labels) = batch

        # Iterate over each image in the batch
        for i in range(len(images)):
            image = images[i]

            # Extract the bounding box for the pedestrian
            cropped_image = extract_bounding_box(
                image, sem_labels[i][0], ins_labels[i][0]
            )

            if cropped_image is not None:
                # Resize the cropped image to 256x256 pixels
                resized_image = torch.nn.functional.interpolate(
                    cropped_image[0].unsqueeze(0), size=(256, 256), mode="bilinear"
                )[0]
                rdebug = torch.nn.functional.interpolate(
                    cropped_image[1].unsqueeze(0).unsqueeze(0),
                    size=(256, 256),
                    mode="bilinear",
                )[0][0]
                torchvision.utils.save_image(resized_image, "build/" + str(I) + ".png")
                #torchvision.utils.save_image(rdebug, "build/" + str(I) + "_y.png")
                I += 1
                print(I)
                if I == 50:
                    quit()
