from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime


class DatasetMetadata(BaseModel):
    name: str
    zip_path: str
    unzipped_path: str
    preprocessed_path: Optional[str]
    data_count: Dict[str, int]
    classes: List[str]
    created_at: datetime
    preprocessed_at: Optional[datetime]
    preprocess_transforms: Optional[Dict[str, str]]


class DatasetListItem(BaseModel):
    id: str
    name: str
    created_at: datetime


class DatasetDetail(BaseModel):
    id: str
    name: str
    zip_path: str
    unzipped_path: Dict[str, Dict[str, str]]
    preprocessed_path: Optional[str]
    data_count: Dict[str, Dict[str, int]]
    classes: List[str]
    created_at: datetime
    preprocessed_at: Optional[datetime]
    preprocessing_steps: List[str]
