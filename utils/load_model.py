from torchvision.models import resnet50


def model(name="resnet50", is_pretrained=False):
    """ Function returns the model used for training

    Args:
        name (str, optional): The name of the model to load. Defaults to "resnet50".
        is_pretrained (bool, optional): Defines if the model is loaded with pretrained weights. Defaults to False.

    Returns:
        torch.model: PyTorch model object
    """
    if name == "resnet50":
        return resnet50(pretrained=is_pretrained)
