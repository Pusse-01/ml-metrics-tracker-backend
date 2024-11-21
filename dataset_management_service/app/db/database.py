from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.MONGODB_URI)
db = client.get_database("ml-metrics-tracker")
