"""Module defines the image augementation 
for the training and test dataset

"""
import torchvision.transforms as transforms


def train_transforms():
    """ Function returns the image transformations for training dataset

    Returns:
        torch.transforms: Transforms Object used to preprocess images during training
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

def val_transforms():
    """ Function returns the image transformations for validation dataset

    Returns:
        torch.transforms: Transforms Object used to preprocess images during validation
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
