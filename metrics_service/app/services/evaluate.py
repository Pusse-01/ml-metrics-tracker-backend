import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import (
    ViTForImageClassification,
    SwinForImageClassification,
    DeiTForImageClassification,
)
import numpy as np
from sklearn.metrics import classification_report
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import DatasetFolder


def find_dataset_directory(root_dir):
    """
    Recursively search for a directory containing 'train','test','valid'.
    Excludes any directories named '__MACOSX' or starting with '.'.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude unwanted directories
        dirnames[:] = [
            d for d in dirnames if not d.startswith("__") and not d.startswith(".")
        ]

        if any(subdir in dirnames for subdir in ["train", "test", "valid"]):
            return dirpath
    raise ValueError(
        "Could not find a dataset directory containing 'train', 'test', or"
        " 'valid' subdirectories."
    )


def load_model(model_id: str):
    """
    Load the specified model.

    Args:
        model_id (str): The model ID ('vit', 'swin', 'deit').

    Returns:
        nn.Module: The loaded model.
    """
    if model_id == "vit":
        return ViTForImageClassification.from_pretrained(
            "app/ml_models/model_weights/vit_base_model"
        )
    elif model_id == "swin":
        return SwinForImageClassification.from_pretrained(
            "app/ml_models/model_weights/swin_transformer_model"
        )
    elif model_id == "deit":
        return DeiTForImageClassification.from_pretrained(
            "app/ml_models/model_weightsdeit_model"
        )
    else:
        raise ValueError("Invalid model_id. Choose from 'vit', 'swin', or 'deit'.")


# class FilteredImageFolder(DatasetFolder):
#     def __init__(self, root, transform=None):
#         super().__init__(
#             root, loader=self.pil_loader, extensions=("jpg", "jpeg", "png")
#         )
#         self.transform = transform

#     @staticmethod
#     def pil_loader(path):
#         try:
#             with open(path, "rb") as f:
#                 img = Image.open(f)
#                 img.verify()  # Verify the file is a valid image
#                 img = Image.open(path)  # Reopen for actual processing
#                 return img.convert("RGB")
#         except (UnidentifiedImageError, OSError):
#             print(f"Skipping invalid or hidden image: {path}")
#             return None

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         # Skip files inside hidden directories like __MACOSX or starting with "._"
#         if "__MACOSX" in path or os.path.basename(path).startswith("._"):
#             print(f"Skipping hidden or invalid image: {path}")
#             return self.__getitem__(
#                 (index + 1) % len(self.samples)
#             )  # Move to next item
#         sample = self.loader(path)
#         if sample is None:  # If the image is invalid, skip this entry
#             print(f"Invalid image file encountered: {path}")
#             return self.__getitem__(
#                 (index + 1) % len(self.samples)
#             )  # Move to next item
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, target

#     def is_valid_file(self, path):
#         # Ensure files are not hidden or in __MACOSX
#         if "__MACOSX" in path or os.path.basename(path).startswith("._"):
#             return False
#         return super().is_valid_file(path)


class FilteredImageFolder(DatasetFolder):
    def __init__(self, root, transform=None):
        super().__init__(
            root, loader=self.pil_loader, extensions=("jpg", "jpeg", "png")
        )
        self.transform = transform
        # Filter samples during initialization
        self.samples = [
            sample for sample in self.samples if self.is_valid_sample(sample[0])
        ]

    @staticmethod
    def pil_loader(path):
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                img.verify()  # Verify that the file is a valid image
                img = Image.open(path)  # Reopen for processing
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"Skipping invalid image: {path}")
            return None

    @staticmethod
    def is_valid_sample(path):
        # Check if the file is valid and not hidden or part of __MACOSX
        if "/__MACOSX/" in path or os.path.basename(path).startswith("._"):
            return False
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.pil_loader(path)
        if sample is None:  # Skip invalid images
            raise IndexError(f"Invalid image file encountered: {path}")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def evaluate_model(model_id: str, dataset_path: str, class_names) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Hemmorrhagic", "Ischemic", "Normal"]

    # Step 1: Load model
    model = load_model(model_id)
    model.to(device).eval()  # Ensure model is on the correct device

    # Step 2: Prepare dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Find the dataset directory dynamically
    data_root_dir = find_dataset_directory(dataset_path)

    # Choose the split you want to evaluate on
    dataset_split = "test"  # or 'train' or 'valid'

    data_dir = os.path.join(data_root_dir, dataset_split)

    # Verify that the directory exists
    if not os.path.isdir(data_dir):
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    dataset = FilteredImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # Map dataset class indices to model class indices
    model_class_indices = {
        class_name: idx for idx, class_name in enumerate(class_names)
    }
    dataset_class_indices = dataset.class_to_idx
    dataset_to_model_label = {}
    for class_name, idx in dataset_class_indices.items():
        if class_name in model_class_indices:
            dataset_to_model_label[idx] = model_class_indices[class_name]
        else:
            print(
                f"Warning: Class '{class_name}' is not in model class indices"
                f" and will be skipped."
            )

    # Step 3: Collect predictions and ground truth
    y_true = []  # Ground truth labels
    y_pred = []  # Model predictions
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Filter out samples with labels not in dataset_to_model_label
            valid_indices = [
                i
                for i, label in enumerate(labels)
                if label.item() in dataset_to_model_label
            ]
            if not valid_indices:
                continue  # Skip if no valid samples in this batch

            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            # Adjust labels to match model's expected class indices
            adjusted_labels = [dataset_to_model_label[label.item()] for label in labels]
            y_true.extend(adjusted_labels)
            y_pred.extend(np.argmax(probs, axis=1))

    # Step 4: Calculate advanced metrics using sklearn
    report = classification_report(
        y_true,
        y_pred,
        labels=list(model_class_indices.values()),
        target_names=class_names,
        output_dict=True,
    )

    # Step 5: Return results
    return {
        "overall_accuracy": report["accuracy"],
        "class_wise_metrics": {
            class_name: metrics
            for class_name, metrics in report.items()
            if class_name in class_names
        },
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }
