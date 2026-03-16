from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as models
modelNames = ["mobilenetV2", "efficientnetB0"]

def freezeModel(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreezeModel(model):
    for param in model.parameters():
        param.requires_grad = True

def buildModel(modelName, numClasses, usePretrained = True, freezeBackbone = True):
    if modelName not in modelNames:
        raise ValueError(f"Model {modelName} is not supported. Please choose from {modelNames}")
    else:
        if modelName == 'mobilenetV2':
            weight = models.MobileNet_V2_Weights.DEFAULT if usePretrained else None
            model = models.mobilenet_v2(weights=weight)
            
            if freezeBackbone:
                freezeModel(model)
                
            model.classifier[1] = nn.Linear(model.last_channel, numClasses)
            return model 

        elif modelName == 'efficientnetB0':
            weight = models.EfficientNet_B0_Weights.DEFAULT if usePretrained else None
            model = models.efficientnet_b0(weights=weight)
            
            if freezeBackbone:
                freezeModel(model)
                
            model.classifier[1] = nn.Linear(model.last_channel, numClasses) 
            return model 