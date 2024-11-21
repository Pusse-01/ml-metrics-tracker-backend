import requests
import os
from urllib.parse import urlparse
import zipfile
from fastapi import HTTPException

API_URL = "http://dataset_management_service:8001"
# API_URL = "http://127.0.0.1:8001"


def get_preprocess_data_url_by_id(dataset_id: str):
    """Fetches the preprocessed data URL for a given dataset ID.

    Args:
        dataset_id (str): The ID of the dataset.

    Returns:
        str: The preprocessed data URL, or None if an error occurs.
    """

    url = f"{API_URL}/datasets/datasets/{dataset_id}/download"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        return data["download_url"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def download_folder_from_s3(dataset_url: str, local_dir: str):
    """
    Download a file from a given URL and save it to a dynamically
    created local directory if not already cached.

    :param dataset_url: URL of the file to download.
    :param local_dir: Base local directory to save the file.
    """
    # Parse components from the URL
    url_parts = urlparse(dataset_url)
    path_parts = url_parts.path.strip("/").split("/")

    # Extract folder name and file name
    folder_name = path_parts[0]
    file_name = path_parts[-1].split("?")[0]

    # Check local cache
    target_dir = os.path.join(local_dir, folder_name)
    local_file_path = os.path.join(target_dir, file_name)

    if os.path.exists(target_dir):
        print(f"Using cached dataset: {target_dir}")
        return local_file_path, target_dir

    # Download the file
    os.makedirs(target_dir, exist_ok=True)
    response = requests.get(dataset_url, stream=True)
    if response.status_code == 200:
        with open(local_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {local_file_path}")
        return local_file_path, target_dir
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to download dataset from {dataset_url}",
        )


def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory.

    :param zip_path: Path to the zip file.
    :param extract_to: Directory where the zip file will be extracted.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"The file {zip_path} is not a valid zip file.")

    # Create target directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Extract zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted contents of {zip_path} to {extract_to}")


def get_class_names(extract_to: str) -> list:
    """
    Get the folder names from the 'test' folder in the extracted directory.

    :param extract_to: Directory where the zip file was extracted.

    Returns:
    List[str]: Folder names inside the 'test' folder.
    """
    # Identify the dataset directory dynamically, ignoring __MACOSX
    extracted_dir = next(
        (
            entry
            for entry in os.listdir(extract_to)
            if os.path.isdir(os.path.join(extract_to, entry)) and entry != "__MACOSX"
        ),
        None,
    )

    if not extracted_dir:
        raise FileNotFoundError("No dataset directory found in the extracted path.")

    test_dir = os.path.join(extract_to, extracted_dir, "test")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"'test' folder not found in path: {test_dir}")

    # List the directories inside the 'test' folder
    class_names = [
        folder
        for folder in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, folder))
    ]

    return class_names


def display_results(results):
    print("\nClassification Results:\n")
    print(f"{'Image':<40} {'Predicted Class':<15} {'Probabilities':<50}")
    print("-" * 110)
    for filename, predicted_class, probabilities in results:
        formatted_probs = ", ".join([f"{p:.2f}" for p in probabilities])
        print(f"{filename:<40} {predicted_class:<15} [{formatted_probs}]")
    print("\n")


def clean_dataset_directory(root_dir):
    """
    Removes all hidden files and directories (like __MACOSX) from the dataset.
    """
    for root, dirs, files in os.walk(root_dir):
        # Remove __MACOSX directories
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        # Remove files starting with ._
        for file in files:
            if file.startswith("._"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed hidden file: {file_path}")
