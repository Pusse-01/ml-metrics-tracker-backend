from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from a `.env` file
load_dotenv()

# MongoDB connection details from environment variables
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Create a MongoDB client
client = MongoClient(MONGO_URI)

# Access the database
db = client[MONGO_DB_NAME]
