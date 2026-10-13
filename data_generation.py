import numpy as np
import torchvision

# Seznam vsech 10 trid v datove sade CIFAR10.
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# Pro nase ucely budou stacit jen 3 tridy.
# Trenovani na vsech 10 by trvalo prilis dlouho.
EASY_CLASSES = [1, 4, 6]  # automobile, deer, frog


def load_cifar10(root='./data', train=True, num_samples=None, seed=42, classes=EASY_CLASSES):
    # Jedna se o malinke obrazky, cela sada se nam v pohode vejde do pameti.
    dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)

    # Obrazky jsou vetsinou ve formatu 0-255 pro kazdy barevny kanal.
    # Tak vysoke hodnoty by nam ale mohly delat problemy se stabilitou treninku,
    # proto je casto normalizujeme do "standardnejsiho" formatu.
    images = dataset.data.astype(np.float64) / 255.0  # (N, 32, 32, 3) uint8 -> [0, 1]
    images = images.transpose(0, 3, 1, 2)             # -> (N, 3, 32, 32), channels-first
    labels = np.array(dataset.targets)

    if classes is not None:
        keep = np.isin(labels, classes)
        images = images[keep]
        labels = labels[keep]
        
        remap = {original: new for new, original in enumerate(classes)}
        labels = np.array([remap[label] for label in labels])

    if num_samples is not None:
        # Datova sada CIFAR10 je pomerne velka, staci nam jen cast...
        indices = np.random.default_rng(seed).choice(len(images), size=num_samples, replace=False)
        images = images[indices]
        labels = labels[indices]

    return list(images), list(labels)


def split_train_val(images, labels, val_fraction=0.1, seed=42):
    num_val = int(len(images) * val_fraction)
    indices = np.random.default_rng(seed).permutation(len(images))
    val_indices, train_indices = indices[:num_val], indices[num_val:]

    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return train_images, train_labels, val_images, val_labels
