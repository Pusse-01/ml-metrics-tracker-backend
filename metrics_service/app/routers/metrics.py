from zipfile import is_zipfile
import os
from fastapi import APIRouter, HTTPException
from app.services.dataset import (
    download_folder_from_s3,
    extract_zip,
    get_class_names,
    get_preprocess_data_url_by_id,
    clean_dataset_directory,
)
from app.services.evaluate import evaluate_model
from app.services.metrics import save_metrics_to_db, get_metrics_by_dataset
from app.models.metrics import MetricData

# from fastapi.encoders import jsonable_encoder
# from app.services.model import classify_images

router = APIRouter()

DATASET_STORAGE_PATH = "app/datasets"


@router.get("/metric-services")
async def metric_services(dataset_id: str, model_type: str):
    # Validate parameters
    if not dataset_id or not model_type:
        raise HTTPException(
            status_code=400,
            detail="Both 'dataset_id' and 'model_type' are required.",
        )

    # Check the local preprocessed folder
    dataset_name = f"{dataset_id}"
    local_dataset_path = os.path.join("preprocessed", dataset_name)
    if os.path.exists(local_dataset_path):
        print(f"Using locally cached dataset: {local_dataset_path}")
        extract_to = os.path.join(local_dataset_path, "extract")
    else:
        # Step 1: Download and extract dataset
        dataset_url = get_preprocess_data_url_by_id(dataset_id)
        if not dataset_url:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset URL not found for ID: {dataset_id}",
            )

        zip_file_path, target_dir = download_folder_from_s3(
            dataset_url,
            "preprocessed",
        )

        # Validate zip file
        if not zip_file_path or not is_zipfile(zip_file_path):
            raise HTTPException(
                status_code=400,
                detail="Downloaded file is not a valid zip file.",
            )

        # Extract zip file
        extract_to = os.path.join(target_dir, "extract")
        extract_zip(zip_file_path, extract_to)
        # Clean unwanted files and directories
        clean_dataset_directory(extract_to)

    # Get class names
    class_names = get_class_names(extract_to)
    if not class_names:
        raise HTTPException(
            status_code=400, detail="No class names found in the dataset."
        )

    # Evaluate model
    results = evaluate_model(model_type, extract_to, class_names)
    return {"results": results}


@router.post("/metrics")
async def save_metrics(metrics: MetricData):
    """
    Save evaluation metrics to the database.

    Args:
        metrics (MetricData): The evaluation metrics.

    Returns:
        dict: The ID of the saved metrics.
    """
    try:
        metrics_id = save_metrics_to_db(metrics.dict())
        return {
            "message": "Metrics saved successfully.",
            "metrics_id": metrics_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_saved_metrics():
    """
    Retrieve saved metrics for a specific dataset.

    Args:
        dataset_id (str): The ID of the dataset.

    Returns:
        dict: The saved metrics or a 404 error if not found.
    """
    try:
        metrics = get_metrics_by_dataset()
        if not metrics:
            raise HTTPException(status_code=404, detail="Metrics not found.")
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
