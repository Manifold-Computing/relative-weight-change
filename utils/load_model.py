from torchvision.models import resnet50


def model(name="resnet50", data='CIFAR10', out_features=10, is_pretrained=False):
    """ Function returns the model used for training

    Args:
        name (str, optional): The name of the model to load. Defaults to "resnet50".
        is_pretrained (bool, optional): Defines if the model is loaded with pretrained weights. Defaults to False.

    Returns:
        torch.model: PyTorch model object
    """
    if name == "resnet50":
        model = resnet50(pretrained=is_pretrained)
        model.fc = nn.Linear(2048, out_features)

        # Preprend a Conv layer to account for 1 input channel
        if data == 'MNIST' or data == 'FMNIST':
            first_conv_layer = [nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3,
                               bias=False)]
            first_conv_layer.extend(list(model.features))  
            model.features= nn.Sequential(*first_conv_layer )
            
        
        return model
