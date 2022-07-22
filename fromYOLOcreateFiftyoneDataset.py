import fiftyone as fo

# A name for the dataset
name = "cubeVision"

# The directory containing the dataset to import
dataset_dir = "yolo"
splits=["val"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
)

# View summary info about the dataset
# print(dataset)
dataset.persistent = True

# Print the first few samples in the dataset
# print(dataset.head())
