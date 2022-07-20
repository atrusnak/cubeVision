import fiftyone as fo

# A name for the dataset
name = "cubeVision"

# The directory containing the dataset to import
dataset_dir = "outputTest"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    data_path="outputTest/data",
    labels_path="outputTest/labels.json",
    dataset_type=dataset_type,
    name=name,
)
session = fo.launch_app(dataset)
