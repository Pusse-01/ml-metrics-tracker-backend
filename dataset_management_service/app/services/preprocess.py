import shutil
import os
from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from typing import Dict, Tuple, Union
from app.dependencies.s3_client import s3_client


def upload_preprocessed_zip_to_s3(
    local_dir: str,
    bucket_name: str,
    s3_prefix: str,
    zip_name: Optional[str] = "preprocessed.zip",
):
    """
    Compresses a local directory into a zip file and uploads it to S3.

    :param local_dir: Path to the local directory to compress and upload
    :param bucket_name: Name of the S3 bucket
    :param s3_prefix: S3 key prefix where the zip file will be uploaded
    :param zip_name: Name of the zip file to create (default: "preprocessed.zip")
    """
    try:
        # Create a zip archive of the local directory
        zip_file_path = os.path.join(local_dir, zip_name or "")
        shutil.make_archive(zip_file_path.replace(".zip", ""), "zip", local_dir)

        # Create the full S3 key for the zip file
        s3_key = os.path.join(s3_prefix, zip_name or "").replace("\\", "/")

        # Upload the zip file to S3
        with open(zip_file_path, "rb") as f:
            s3_client.upload_fileobj(f, bucket_name, s3_key)

        print(f"Uploaded {zip_file_path} to s3://{bucket_name}/{s3_key}")

    except Exception as e:
        print(f"Error compressing or uploading preprocessed data: {e}")
        raise

    finally:
        # Clean up the zip file after upload
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)


def get_transforms(
    options: Dict[str, Union[bool, Tuple[int, int], int, float]]
) -> transforms.Compose:
    transform_list = []

    # Resize if specified
    if options.get("resize"):
        transform_list.append(transforms.Resize(options["resize"]))

    # Apply grayscale transformation if requested
    if options.get("grayscale"):
        transform_list.append(transforms.Grayscale(num_output_channels=1))
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Convert to tensor before normalization
    transform_list.append(transforms.ToTensor())

    # Apply normalization only if normalize=True
    if options.get("normalize", False):  # Default to False if normalize not specified
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    # Apply rotation if rotation is specified and not zero
    rotation = options.get("rotation", 0)
    if rotation and rotation != 0:
        transform_list.append(transforms.RandomRotation(rotation))

    # Apply horizontal flip if requested
    if options.get("horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())

    # Apply vertical flip if requested
    if options.get("vertical_flip", False):
        transform_list.append(transforms.RandomVerticalFlip())

    # Apply ColorJitter only if the values are not neutral
    brightness = options.get("brightness", 1)
    contrast = options.get("contrast", 1)
    saturation = options.get("saturation", 1)
    hue = options.get("hue", 0)

    if brightness != 1 or contrast != 1 or saturation != 1 or hue != 0:
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    return transforms.Compose(transform_list)


def find_dataset_root(input_dir: str) -> str:
    """Finds the correct root directory that contains 'train','valid','test' folders."""
    for root, dirs, files in os.walk(input_dir):
        if {"train", "valid", "test"}.issubset(set(dirs)):
            # print(f"Found dataset root directory: {root}")
            return root
    raise FileNotFoundError(
        "Dataset root with 'train', 'valid', and 'test' directories not found."
    )


def preprocess_images(
    input_dir: str,
    output_dir: str,
    options: Dict[str, Union[bool, Tuple[int, int], int, float]],
) -> Dict[str, Dict[str, list]]:
    os.makedirs(output_dir, exist_ok=True)
    dataset_structure: Dict[str, Dict[str, list]] = {}

    # Locate the actual dataset root directory containing 'train', 'valid', and 'test'
    dataset_root = find_dataset_root(input_dir)

    transform = get_transforms(options)
    # print(f"Using transform: {transform}")

    for split in ["train", "valid", "test"]:
        split_path = os.path.join(dataset_root, split)
        # print(f"Checking directory: {split_path}")

        if not os.path.exists(split_path):
            # print(f"Directory does not exist: {split_path}")
            continue

        dataset_structure[split] = {}
        split_preprocess_path = os.path.join(output_dir, split)
        os.makedirs(split_preprocess_path, exist_ok=True)
        # print(f"Split output path created: {split_preprocess_path}")

        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            # print(f"Checking class folder: {class_path}")

            if not os.path.isdir(class_path):
                # print(f"Not a directory: {class_path}")
                continue

            dataset_structure[split][class_folder] = []
            class_preprocess_path = os.path.join(split_preprocess_path, class_folder)
            os.makedirs(class_preprocess_path, exist_ok=True)
            # print(f"Class output path created: {class_preprocess_path}")

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                # print(f"Processing image: {img_path}")

                # Skip non-image files like .DS_Store
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")

                        img_transformed = transform(img)
                        # print(f"Transformed image tensor: {img_transformed}")

                        if isinstance(img_transformed, torch.Tensor):
                            img_transformed = transforms.ToPILImage()(img_transformed)

                        save_path = os.path.join(class_preprocess_path, img_file)
                        img_transformed.save(save_path)
                        dataset_structure[split][class_folder].append(img_file)
                        # print(f"Saved transformed image to: {save_path}")

                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue

    # print(f"Final dataset structure: {dataset_structure}")
    return dataset_structure
