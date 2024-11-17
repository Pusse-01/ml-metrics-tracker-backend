import os
import uuid
from botocore.exceptions import NoCredentialsError
from fastapi import HTTPException
import zipfile
from app.dependencies.s3_client import s3_client
from app.core.config import settings
from app.db.database import db
from typing import Tuple
from bson import ObjectId
from botocore.exceptions import ClientError


def serialize_document(doc):
    """Convert MongoDB document to a JSON serializable format."""
    if not doc:
        return None
    doc["id"] = str(doc.pop("_id"))  # Convert ObjectId and rename _id to id
    return doc


def upload_file_to_s3(file_obj, bucket_name, key):
    s3_client.upload_fileobj(file_obj, bucket_name, key)
    return f"s3://{bucket_name}/{key}"


def validate_and_upload_zip(file_obj, dataset_name: str):
    zip_key = f"{dataset_name}/original.zip"
    zip_path_s3 = upload_file_to_s3(file_obj, settings.S3_BUCKET_NAME, zip_key)

    # Further validation or extraction logic can go here if needed
    return zip_path_s3


def check_dataset_exists(dataset_name):
    """Check if a dataset folder exists in MongoDB."""
    # Generate a unique folder name if dataset_name is not provided
    if dataset_name:
        # Check if dataset name already exists in MongoDB
        existing_dataset = db.datasets.find_one({"name": dataset_name})
        if existing_dataset:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset with name '{dataset_name}' already exists.",
            )
        folder_name = dataset_name
    else:
        folder_name = str(uuid.uuid4())
    return folder_name


def upload_dataset_to_s3(zip_path, temp_dir, folder_name, dataset_structure):
    """Upload the dataset to S3 with both zipped and unzipped versions."""
    # Upload zip file to S3
    zip_s3_key = f"{folder_name}/original.zip"
    with open(zip_path, "rb") as zip_file_obj:
        zip_path_s3 = upload_file_to_s3(
            zip_file_obj, settings.S3_BUCKET_NAME, zip_s3_key
        )

    # Upload unzipped files to S3 by class
    extracted_s3_paths = {}
    for split, classes in dataset_structure.items():
        split_s3_path = f"{folder_name}/unzipped/{split}/"
        extracted_s3_paths[split] = {}

        for class_folder, files in classes.items():
            class_s3_path = f"{split_s3_path}{class_folder}/"
            extracted_s3_paths[split][class_folder] = class_s3_path

            for image_file in files:
                file_path = os.path.join(temp_dir, split, class_folder, image_file)
                file_key = f"{class_s3_path}{image_file}"
                with open(file_path, "rb") as img_file:
                    upload_file_to_s3(img_file, settings.S3_BUCKET_NAME, file_key)

    return {"zip": zip_path_s3, "unzipped": extracted_s3_paths}


def validate_and_extract_zip(zip_path, extract_to):
    """
    Validates and extracts a zip file, ensuring it contains 'train', 'test',
    and 'valid' folders with images in png/jpg format, organized by class folders.
    :param zip_path: Path to the zip file
    :param extract_to: Directory where the zip should be extracted
    :return: Dictionary with the structure of the dataset
    """
    required_dirs = {"train", "valid", "test"}
    valid_extensions = {".png", ".jpg", ".jpeg"}
    dataset_structure = {}

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_dirs = {
            os.path.split(item.filename)[0]
            for item in zip_ref.infolist()
            if item.is_dir()
        }

        # Find the base directory that may contain 'train', 'valid',
        # 'test' subdirectories
        base_dir = None
        for dir_name in extracted_dirs:
            sub_dirs = {
                os.path.basename(d) for d in extracted_dirs if d.startswith(dir_name)
            }
            if required_dirs.issubset(sub_dirs):
                base_dir = dir_name
                break

        if base_dir is None:
            raise HTTPException(
                status_code=400,
                detail="Zip file must contain 'train', 'test', 'valid' directories.",
            )

        # Validate file types and collect class structure
        for required_dir in required_dirs:
            dir_path = os.path.join(extract_to, base_dir, required_dir)
            if not os.path.exists(dir_path):
                raise HTTPException(
                    status_code=400, detail=f"Directory '{required_dir}' is missing."
                )

            # Initialize structure for this main directory
            dataset_structure[required_dir] = {}

            # Check for class subdirectories inside the required directories
            for class_folder in os.listdir(dir_path):
                class_path = os.path.join(dir_path, class_folder)
                if os.path.isdir(class_path):
                    dataset_structure[required_dir][class_folder] = []
                    for file_name in os.listdir(class_path):
                        file_ext = os.path.splitext(file_name)[1].lower()
                        if file_ext in valid_extensions:
                            dataset_structure[required_dir][class_folder].append(
                                file_name
                            )
                        elif file_name == ".DS_Store":
                            continue
                        else:
                            raise HTTPException(
                                status_code=400,
                                detail="Only png and jpg images are allowed.",
                            )
                else:
                    if class_folder != ".DS_Store":
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Expected a directory for each class inside "
                                f"'{required_dir}', "
                                f"found file '{class_folder}' instead."
                            ),
                        )

    return dataset_structure, base_dir


# def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
#     """
#     Parses an S3 URI and returns the bucket name and key.

#     :param s3_uri: S3 URI in the format s3://bucket_name/key
#     :return: Tuple of bucket name and key
#     """
#     if s3_uri.startswith("s3://"):
#         s3_uri = s3_uri[5:]
#         bucket_name, key = s3_uri.split("/", 1)
#         return bucket_name, key
#     else:
#         raise ValueError("Invalid S3 URI")


def get_all_datasets():
    """Fetch all datasets from the database."""
    datasets = db.datasets.find()
    return [serialize_document(dataset) for dataset in datasets]


# def get_dataset_by_id(dataset_id: str):
#     """Fetch dataset details by ID."""
#     if not ObjectId.is_valid(dataset_id):
#         raise HTTPException(status_code=400, detail="Invalid dataset ID format.")

#     dataset = db.datasets.find_one({"_id": ObjectId(dataset_id)})
#     if not dataset:
#         raise HTTPException(status_code=404, detail="Dataset not found.")

#     return dataset


def download_preprocessed_folder(dataset_id: str, local_dir: str):
    """Download preprocessed folder from S3."""
    dataset = get_dataset_by_id(dataset_id)
    preprocessed_path = dataset.get("preprocessed_path")
    if not preprocessed_path:
        raise HTTPException(
            status_code=404, detail="Preprocessed data path not found in the dataset."
        )

    bucket_name, s3_prefix = preprocessed_path[5:].split("/", 1)
    download_folder_from_s3(bucket_name, s3_prefix, local_dir)
    return local_dir


def download_folder_from_s3(bucket_name: str, prefix: str, local_dir: str):
    """
    Download a folder from S3 to a local directory.

    :param bucket_name: Name of the S3 bucket
    :param prefix: S3 folder prefix
    :param local_dir: Local directory to save the files
    """
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                file_key = obj["Key"]
                local_file_path = os.path.join(
                    local_dir, os.path.relpath(file_key, prefix)
                )
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                try:
                    s3_client.download_file(bucket_name, file_key, local_file_path)
                except ClientError as e:
                    raise Exception(f"Failed to download {file_key}: {str(e)}")


def generate_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600):
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
        return url
    except NoCredentialsError as e:
        raise RuntimeError("AWS credentials not configured") from e


def get_dataset_by_id(dataset_id: str):
    """
    Fetch a dataset by its ID from the database.

    Args:
        dataset_id (str): The ID of the dataset to retrieve.

    Returns:
        dict: The dataset document from the database.

    Raises:
        HTTPException: If the dataset ID is invalid or the dataset is not found.
    """
    # Validate the ObjectId format
    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(status_code=400, detail="Invalid dataset ID format.")

    # Fetch the dataset from the database
    dataset = db.datasets.find_one({"_id": ObjectId(dataset_id)})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return dataset


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parses an S3 URI and returns the bucket name and object key.

    Args:
        s3_uri (str): The S3 URI to parse (e.g., "s3://bucket_name/key").

    Returns:
        Tuple[str, str]: A tuple containing the bucket name and object key.

    Raises:
        ValueError: If the S3 URI is invalid.
    """
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]  # Remove the "s3://" prefix
        bucket_name, key = s3_uri.split("/", 1)
        return bucket_name, key
    else:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
