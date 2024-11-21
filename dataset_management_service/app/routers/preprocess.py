from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from bson import ObjectId
import os
import zipfile
import datetime
import tempfile
from app.db.database import db
from app.dependencies.s3_client import s3_client
from app.services.dataset import parse_s3_uri
from app.services.preprocess import preprocess_images, upload_preprocessed_zip_to_s3

# from app.services.dataset import validate_and_extract_zip

router = APIRouter()

DATASET_STORAGE_PATH = "app/datasets"


@router.post("/preprocess-dataset")
async def upload_and_preprocess_dataset(
    dataset_id: str,
    dataset_name: Optional[str] = Form(None),
    resize_width: Optional[int] = Form(None),
    resize_height: Optional[int] = Form(None),
    grayscale: bool = Form(False),
    normalize: bool = Form(False),
    rotation: Optional[int] = Form(None),
    horizontal_flip: bool = Form(False),
    vertical_flip: bool = Form(False),
    brightness: Optional[float] = Form(None),
    contrast: Optional[float] = Form(None),
    saturation: Optional[float] = Form(None),
    hue: Optional[float] = Form(None),
):
    """
    Uploads a dataset, extracts and validates its contents,
    applies user-selected preprocessing,
    and saves the preprocessed images in a structured folder.

    Parameters:
    - file: Zip file containing the dataset.
    - dataset_name: Optional name for the dataset folder.
    - resize: Tuple for resizing images (width, height).
    - grayscale: Convert images to grayscale if True.
    - normalize: Apply normalization if True.
    - rotation: Integer for random rotation in degrees.
    - horizontal_flip: Apply horizontal flip if True.
    - vertical_flip: Apply vertical flip if True.
    - brightness: Float for brightness adjustment.
    - contrast: Float for contrast adjustment.
    - saturation: Float for saturation adjustment.
    - hue: Float for hue adjustment.
    """

    # Construct resize tuple if width and height are provided
    resize = (resize_width, resize_height) if resize_width and resize_height else None

    # Define preprocessing options
    preprocessing_steps = {
        "resize": resize,
        "grayscale": grayscale,
        "normalize": normalize,
        "rotation": rotation if rotation is not None else 0,  # Default to 0 if None
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "brightness": (
            brightness if brightness is not None else 1.0
        ),  # Default to 1.0 (no change)
        "contrast": (
            contrast if contrast is not None else 1.0
        ),  # Default to 1.0 (no change)
        "saturation": (
            saturation if saturation is not None else 1.0
        ),  # Default to 1.0 (no change)
        "hue": hue if hue is not None else 0,  # Default to 0 (no hue change)
    }

    # Step 1: Retrieve dataset metadata from MongoDB using dataset_id
    try:
        dataset = db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format.")

    # Step 2: Get the zip file path from dataset metadata
    zip_s3_path = dataset.get("zip_path")
    if not zip_s3_path:
        raise HTTPException(
            status_code=500, detail="Zip path not found in dataset metadata."
        )

    # Step 3: Download the zip file from S3
    bucket_name, zip_s3_key = parse_s3_uri(zip_s3_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "dataset.zip")
        try:
            s3_client.download_file(bucket_name, zip_s3_key, zip_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to download dataset from S3: {str(e)}"
            )

        # Step 4: Extract zip file
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file.")

        # Step 5: Define preprocessing options
        resize = (
            (resize_width, resize_height) if resize_width and resize_height else None
        )

        # Step 6: Preprocess images
        input_dir = temp_dir
        preprocess_output_dir = os.path.join(temp_dir, "preprocessed")
        try:
            processed_structure = preprocess_images(
                input_dir, preprocess_output_dir, preprocessing_steps
            )
            print(processed_structure)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to preprocess images: {str(e)}"
            )

        # Step 7: Upload preprocessed images to S3
        folder_name = dataset["name"]
        preprocessed_s3_key_prefix = f"{folder_name}/preprocessed/"
        try:
            upload_preprocessed_zip_to_s3(
                preprocess_output_dir, bucket_name, preprocessed_s3_key_prefix
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload preprocessed data to S3: {str(e)}",
            )

    # Step 8: Update dataset metadata
    preprocessed_at = datetime.datetime.utcnow().isoformat()
    preprocessed_s3_path = (
        f"s3://{bucket_name}/{preprocessed_s3_key_prefix}preprocessed.zip"
    )
    db.datasets.update_one(
        {"_id": ObjectId(dataset_id)},
        {
            "$set": {
                "preprocessed_at": preprocessed_at,
                "preprocessing_steps": preprocessing_steps,
                "preprocessed_path": preprocessed_s3_path,
            }
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Dataset preprocessed and uploaded successfully.",
            "dataset_id": dataset_id,
            "preprocessed_at": preprocessed_at,
            "preprocessing_steps": preprocessing_steps,
        },
    )
