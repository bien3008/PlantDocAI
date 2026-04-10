import torch
import torch.nn as nn
import timm

def freezeModel(model):
    """Utility to freeze all parameters in a generic torch module."""
    for param in model.parameters():
        param.requires_grad = False

def unfreezeModel(model):
    """Utility to unfreeze all parameters in a generic torch module."""
    for param in model.parameters():
        param.requires_grad = True

def buildModel(modelName: str, numClasses: int, usePretrained: bool = True, freezeBackbone: bool = True):
    """
    Builds a PyTorch image classification model using `timm`.
    
    Args:
        modelName (str): Name of the architecture (e.g., 'mobilenetv2_100', 'resnet50', 'convnext_tiny').
        numClasses (int): Number of target classes for the final classification layer.
        usePretrained (bool): If True, loads pre-trained ImageNet weights.
        freezeBackbone (bool): If True, freezes all layers except the newly initialized classifier head.
        
    Returns:
        torch.nn.Module: The configured PyTorch model ready for training.
    """
    try:
        # timm automatically handles replacing the final classifier head to match num_classes
        model = timm.create_model(modelName, pretrained=usePretrained, num_classes=numClasses)
    except Exception as e:
        raise ValueError(f"Failed to create model '{modelName}' using timm. Is the model name valid in timm? Error: {e}")

    if freezeBackbone:
        # Freeze the entire model originally
        freezeModel(model)
        
        # Then safely unfreeze only the classifier head
        classifier = model.get_classifier()
        if classifier is not None:
            unfreezeModel(classifier)
            
    return model