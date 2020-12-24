import torchvision.transforms as transforms


def train_transforms():
    """ Function returns the image transformations for training dataset

    Returns:
        torch.transforms: Transforms Object used to preprocess images during training
    """

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

    return train_transforms

def val_transforms():
    """ Function returns the image transformations for validation dataset

    Returns:
        torch.transforms: Transforms Object used to preprocess images during validation
    """

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
    return val_transforms
