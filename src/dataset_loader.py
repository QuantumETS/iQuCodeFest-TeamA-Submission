import os
import numpy as np
from PIL import Image
import torch


CLASSES = {"glioma_tumor": 0, "pituitary_tumor": 1, "normal": 2, "meningioma_tumor": 3}

IMAGE_SIZE = (256, 256)


def load_brain_tumor_dataset(dataset_path, grayscale=True, normalize=True):
    """
    Loads a brain tumor image dataset into NumPy arrays.

    Expected folder structure:

    dataset/
    ├── glioma_tumor/
    ├── pituitary_tumor/
    ├── normal/
    └── meningioma_tumor/

    Returns:
        X: np.ndarray of images
        y: np.ndarray of labels
    """

    images = []
    labels = []

    for class_name, label in CLASSES.items():
        class_folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_folder):
            print(f"Warning: folder not found: {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(class_folder, filename)

                try:
                    image = Image.open(image_path)

                    if grayscale:
                        image = image.convert("L")
                    else:
                        image = image.convert("RGB")

                    image = image.resize(IMAGE_SIZE)

                    image_array = np.array(image, dtype=np.float32)

                    if normalize:
                        image_array = image_array / 255.0

                    images.append(image_array)
                    labels.append(label)

                except Exception as e:
                    print(f"Error loading {image_path}: {e}")

    X = np.array(images)
    y = np.array(labels)

    if grayscale:
        # CNN format: (number_of_images, 256, 256, 1)
        X = np.expand_dims(X, axis=-1)

    return X, y


def dataset_to_torch(X, y):
    """
    Converts the dataset from NumPy arrays to PyTorch tensors.

    Args:
        X: np.ndarray of images
        y: np.ndarray of labels

    Returns:
        X_tensor: torch.Tensor of images
        y_tensor: torch.Tensor of labels
    """
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor


if __name__ == "__main__":
    dataset_path = "data/archive/Data"
    X, y = load_brain_tumor_dataset(dataset_path)
    print("Dataset loaded:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_tensor, y_tensor = dataset_to_torch(X, y)
    print("Tensors created:")
    print("X_tensor shape:", X_tensor.shape)
    print("y_tensor shape:", y_tensor.shape)
