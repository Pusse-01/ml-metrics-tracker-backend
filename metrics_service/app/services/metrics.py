import datetime
from app.db.database import db
from pymongo.errors import PyMongoError


def save_metrics_to_db(metrics: dict):
    """
    Save the evaluation metrics to the database.

    Args:
        metrics (dict): The evaluation metrics to save.

    Returns:
        str: The ID of the inserted document.
    """
    metrics["created_at"] = datetime.datetime.utcnow().isoformat()
    try:
        result = db.metrics.insert_one(metrics)
        return str(result.inserted_id)
    except PyMongoError as e:
        raise Exception(f"Failed to save metrics: {e}")


def get_metrics_by_dataset():
    """
    Retrieve saved metrics for a specific dataset from the database.

    Args:
        dataset_id (str): The ID of the dataset.

    Returns:
        dict: The saved metrics or None if not found.
    """
    try:
        metrics = db.metrics.find()
        return list(metrics)
    except PyMongoError as e:
        raise Exception(f"Failed to retrieve metrics: {e}")
