from dataset_loader import dataset_separator, load_brain_tumor_dataset


DATASET_PATH = "data/archive/Data"


images, labels = load_brain_tumor_dataset(DATASET_PATH)
# images[0]  # the pixels of the first brain MRI image
# labels[0]  # the answer/class for that image

print("images shape:", images.shape)
print("labels shape:", labels.shape)

print("First image shape:", images[0].shape)
print("First label:", labels[0])

X_train, y_train, X_test, y_test = dataset_separator(images, labels)
print("Training set shapes:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Testing set shapes:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)