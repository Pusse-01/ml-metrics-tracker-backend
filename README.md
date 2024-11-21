# ML Metrics Tracker

A microservices-based application for managing datasets, evaluating machine learning models, and tracking performance metrics over time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Environment Configuration](#environment-configuration)
- [Services Overview](#services-overview)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Known Issues](#known-issues)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

The **ML Metrics Tracker** application provides functionality to:
- Upload and preprocess datasets.
- Train machine learning models and evaluate their performance.
- Store evaluation metrics for historical tracking.
- Retrieve and display metrics via a web-based interface.

---

## Features

- **Dataset Management**:
  - Upload datasets (ZIP format).
  - Preprocess datasets with options like resizing, normalization, and augmentation.

- **Metrics Evaluation**:
  - Compute metrics like accuracy, precision, recall, and F1 scores.
  - Save metrics to a database for historical tracking.
  
- **Web Interface**:
  - Built with Streamlit to allow users to interact with the application easily.

- **Microservices Architecture**:
  - Independent services for Dataset Management and Metrics Evaluation.
  - Connected through Traefik for API Gateway functionality.

---

## Technologies Used

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Databases**: MongoDB, AWS S3
- **Containerization**: Docker, Docker Compose
- **API Gateway**: Traefik
- **Model Frameworks**: PyTorch, Transformers (ViT, Swin, DeIT)
- **Pre-Commit Hooks**: Code formatting and quality checks

---

## Getting Started

### Prerequisites

Before starting, ensure you have the following installed:
- [Python 3.11+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/)
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (or local MongoDB instance)
- AWS S3 bucket for dataset storage.

---

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ml-metrics-tracker.git
   cd ml-metrics-tracker
   ```
Set Up Virtual Environment (optional):

```bash 
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
```
Install Dependencies:
```bash
pip install -r requirements.txt
```
Configure Environment Variables: See the Environment Configuration section.

Running the Application
Build and Start Services:
docker-compose up --build
Access the Web Interface: Open your browser and navigate to:
Streamlit App: http://localhost:8501
Access Swagger Docs:
Dataset Management Service: http://localhost/dataset/docs
Metrics Service: http://localhost/metrics/docs
Environment Configuration

Create two .env files in the root directory: .env.dataset_management_service and .env.metrics_service.

.env.dataset_management_service
PROJECT_NAME=ml-metrics-tracker-dataset-management
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=ml-metrics-tracker
S3_BUCKET_NAME=ml-metrics-tracker
AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
AWS_REGION=us-east-1
.env.metrics_service
PROJECT_NAME=ml-metrics-tracker-metrics
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=ml-metrics-tracker
S3_BUCKET_NAME=ml-metrics-tracker
AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
AWS_REGION=us-east-1
Services Overview

Dataset Management Service
Handles:

Dataset upload and validation.
Dataset preprocessing.
Data storage in MongoDB and AWS S3.
Metrics Service
Handles:

Model evaluation (e.g., accuracy, F1 score).
Metrics storage in MongoDB.
Metrics retrieval.
API Endpoints

Service	Endpoint	Method	Description
Dataset Management	/datasets/upload	POST	Upload a new dataset.
/datasets/preprocess	POST	Preprocess a dataset.
Metrics Service	/metrics/evaluate	GET	Evaluate a model's metrics.
/metrics/save	POST	Save evaluated metrics to the database.
/metrics/retrieve	GET	Retrieve saved metrics.
Testing

Testing with pytest is not yet implemented. Manual testing can be done via Postman or Curl.

Known Issues

Shared Memory Limitations: Resolved by setting shm_size: "256m" in docker-compose.yml.
Environment Consistency: Resolved by synchronizing requirements.txt across environments.
Future Enhancements

Add pytest for automated testing.
Optimize the dataset preprocessing pipeline.
Enhance the web interface for better user experience.
License

This project is licensed under the MIT License. See the LICENSE file for details.


This `README.md` is structured to provide comprehensive instructions and necessary details for running and maintaining the project. It can be directly added to your repository.