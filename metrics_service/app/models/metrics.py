from pydantic import BaseModel
from typing import Dict, Any


class MetricData(BaseModel):
    dataset_id: str
    model_type: str
    overall_accuracy: float
    macro_avg: Dict[str, Any]
    weighted_avg: Dict[str, Any]
    class_wise_metrics: Dict[str, Any]
    created_at: str
