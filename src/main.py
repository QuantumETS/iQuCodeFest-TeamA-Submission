from dataset_loader import load_brain_tumor_dataset


DATASET_PATH = "data/archive/Data"


images, labels = load_brain_tumor_dataset(DATASET_PATH)
# images[0]  # the pixels of the first brain MRI image
# labels[0]  # the answer/class for that image

print("images shape:", images.shape)
print("labels shape:", labels.shape)

print("First image shape:", images[0].shape)
print("First label:", labels[0])