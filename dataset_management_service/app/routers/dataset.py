from fastapi import APIRouter, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional  # Dict, List,
import shutil
import os
import datetime
import tempfile
from app.services.dataset import (
    check_dataset_exists,
    validate_and_extract_zip,
    upload_file_to_s3,
)
from app.core.config import settings
from app.db.database import db

# import concurrent.futures
from app.services.dataset import (
    get_all_datasets,
    generate_presigned_url,
    parse_s3_uri,
    get_dataset_by_id,
)

router = APIRouter()

DATASET_STORAGE_PATH = "app/datasets"


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile, dataset_name: Optional[str] = Form(None)):
    """
    Uploads a dataset zip file, validates its contents, and stores metadata.

    Args:
        file (UploadFile): The uploaded zip file containing the dataset.
        dataset_name (Optional[str], optional): The desired name for the
        dataset.
            If not provided, a unique name will be generated.

    Returns:
        JSONResponse: A JSON response containing a success message, metadata
        ID, and metadata details.

    Raises:
        HTTPException: Various exceptions depending on failure points, such as
        invalid file type,
            failed validation, or database errors.
    """
    # Step 1: Validate file type
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only zip files are allowed.",
        )

    # Generate a unique folder name if dataset_name is not provided
    folder_name = check_dataset_exists(dataset_name)
    file_name = file.filename

    # Step 2: Save the uploaded file temporarily for validation
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, file_name)

        # Save the file locally in the temp directory
        try:
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save zip file locally: {str(e)}",
            )

        # Step 3: Validate and extract zip contents
        try:
            dataset_structure, base_dir = validate_and_extract_zip(zip_path, temp_dir)
        except HTTPException as e:
            raise e  # Return HTTPException directly if validation fails
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to validate zip contents: {str(e)}",
            )

        # Step 4: Upload the validated zip and extracted folders to S3
        try:
            # Upload the original zip file to S3 under the dataset folder
            zip_s3_key = f"{folder_name}/{file_name}.zip"
            with open(zip_path, "rb") as zip_file_obj:
                zip_path_s3 = upload_file_to_s3(
                    zip_file_obj, settings.S3_BUCKET_NAME, zip_s3_key
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload dataset to S3: {str(e)}",
            )

    # Step 5: Save metadata to MongoDB
    data_count = {
        split: {
            class_name: len(files)
            for class_name, files in dataset_structure[split].items()
        }
        for split in dataset_structure
    }

    classes = list({cls for split in dataset_structure.values() for cls in split})
    unzipped_s3_path = f"s3://{settings.S3_BUCKET_NAME}/{folder_name}/unzipped/"

    metadata = {
        "name": folder_name,
        "zip_path": zip_path_s3,
        "unzipped_path": unzipped_s3_path,
        "unzipped_folder_structure": "",
        "data_count": data_count,
        "classes": classes,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "preprocessed_at": None,
        "preprocessing_steps": [],
    }

    try:
        result = db.datasets.insert_one(metadata)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save metadata to MongoDB: {str(e)}"
        )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Dataset uploaded, validated, and metadata saved successfully.",
            "metadata_id": str(result.inserted_id),
            "metadata": str(metadata),
        },
    )


@router.get("/datasets")
async def list_datasets():
    """
    Retrieves a list of all available datasets.

    Returns:
        List[dict]: A list of dataset metadata dictionaries.

    Raises:
        HTTPException: If no datasets are found or if an error occurs during retrieval.
    """
    try:
        datasets = get_all_datasets()
        print(datasets)
        if not datasets:
            raise HTTPException(status_code=404, detail="No datasets found.")
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """
    Generates a presigned URL for downloading the specified dataset.

    Args:
        dataset_id (str): The unique identifier of the dataset to download.

    Returns:
        dict: A dictionary containing the presigned download URL.

    Raises:
        HTTPException: If the dataset is not found or if neither the
        preprocessed nor the zip path is available.
    """

    # Fetch dataset by ID
    dataset = get_dataset_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Check for preprocessed path or fallback to zip path
    preprocessed_path = dataset.get("preprocessed_path")
    zip_path = dataset.get("zip_path")

    if preprocessed_path:
        # Generate presigned URL for preprocessed path
        bucket_name, object_key = parse_s3_uri(preprocessed_path)
    elif zip_path:
        # Generate presigned URL for zip path
        bucket_name, object_key = parse_s3_uri(zip_path)
    else:
        # Return error if neither is available
        raise HTTPException(
            status_code=404, detail="Dataset or preprocessed data not found."
        )

    # Generate presigned URL
    presigned_url = generate_presigned_url(bucket_name, object_key)

    return {"download_url": presigned_url}
