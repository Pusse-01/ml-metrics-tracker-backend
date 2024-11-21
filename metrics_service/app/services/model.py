# import os
# import torch
# from PIL import Image
# import numpy as np
# from torchvision import transforms

# # from transformers import (
# #     # ViTForImageClassification,
# #     # SwinForImageClassification,
# #     # DeiTForImageClassification,
# # )
# from torch import nn

# # Define class names
# # class_names = ["Hemmorrhagic", "Ischemic", "Normal"]

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define image transformation
# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )


# # Define EnsembleModel class
# class EnsembleModel(nn.Module):
#     def __init__(self, models):
#         super(EnsembleModel, self).__init__()
#         self.models = models

#     def forward(self, x):
#         outputs = [
#             model(x).logits if hasattr(model(x), "logits") else model(x)
#             for model in self.models
#         ]
#         output = torch.mean(torch.stack(outputs), dim=0)
#         return output


# # def load_models():
# #     """Load all required models with debug prints."""
# #     print("[DEBUG] Loading models...")
# #     try:
# #         # vit_model = ViTForImageClassification.from_pretrained
# # ("./models/vit_base_model")
# #         swin_model = SwinForImageClassification.from_pretrained(
# #             "app/ml_models/model_weights/swin_transformer_model"
# #         )
# #         # deit_model = DeiTForImageClassification.from_pretrained
# # ("./models/deit_model")

# #         # Print model loading status
# #         print("[DEBUG] Swin model loaded successfully.")

# #         # Move models to device
# #         # vit_model.to(device).eval()
# #         swin_model.to(device).eval()
# #         print("[DEBUG] Swin model moved to device.")

# #         # deit_model.to(device).eval()
# #         ensemble_model = EnsembleModel([swin_model]).to(device).eval()
# #         print("[DEBUG] Ensemble model created successfully.")
# #         return swin_model, ensemble_model
# #     except Exception as e:
# #         print(f"[ERROR] Failed to load models: {e}")
# #         raise


# def load_images_from_folder(folder_path):
#     """Load all images from the specified folder with debug prints."""
#     print(f"[DEBUG] Loading images from folder: {folder_path}")
#     images = []
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             image = Image.open(file_path).convert("RGB")
#             images.append((filename, image))
#             print(f"[DEBUG] Successfully loaded image: {filename}")
#         except Exception as e:
#             print(f"[ERROR] Failed to load image {filename}: {e}")
#     print(f"[DEBUG] Total images loaded: {len(images)}")
#     return images


# def predict_image(image, model, is_ensemble=False):
#     """Predict the class probabilities for an image with debug prints."""
#     print(f"[DEBUG] Predicting image...")
#     try:
#         image_tensor = data_transforms(image).unsqueeze(0).to(device)
#         print(f"[DEBUG] Image transformed and moved to device.")

#         with torch.no_grad():
#             output = model(image_tensor)
#             logits = output if is_ensemble else output.logits
#             probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         print(f"[DEBUG] Prediction successful. Probabilities: {probs}")
#         return probs
#     except Exception as e:
#         print(f"[ERROR] Failed to predict image: {e}")
#         raise


# def classify_images(folder_path, model_choice, class_names):
#     """
#     Classify a set of images using the specified model with debug prints.

#     Parameters:
#         folder_path (str): Path to the folder containing images.
#         model_choice (str): Model to use for classification ('ViT', 'Swin',
# 'DeiT', or 'Ensemble').

#     Returns:
#         List of tuples: Each tuple contains (filename, predicted_class,
# probabilities).
#     """
#     print(f"[DEBUG] Starting classification with model: {model_choice}")

#     # Load models
#     swin_model, ensemble_model = load_models()

#     # Map model choice to actual model
#     model_mapping = {
#         "Swin": swin_model,
#         "Ensemble": ensemble_model,
#     }

#     # Ensure valid model choice
#     model = model_mapping.get(model_choice)
#     if not model:
#         raise ValueError(f"[ERROR] Invalid model choice: {model_choice}")

#     # Load images
#     images = load_images_from_folder(folder_path)
#     if not images:
#         raise ValueError(f"[ERROR] No valid images found in folder: {folder_path}")

#     # Process images
#     results = []
#     for filename, image in images:
#         print(f"[DEBUG] Classifying image: {filename}")
#         try:
#             probs = predict_image(
#                 image, model, is_ensemble=(model_choice == "Ensemble")
#             )
#             pred_class = class_names[np.argmax(probs)]
#             print(f"[DEBUG] Image: {filename}, Predicted class: {pred_class}")
#             results.append((filename, pred_class, probs))
#         except Exception as e:
#             print(f"[ERROR] Failed to classify image {filename}: {e}")

#     print(f"[DEBUG] Classification complete. Total images classified: {len(results)}")
#     return results
