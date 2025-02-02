# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from features.resnet import resnet18, resnet18_84
from features.efficientnet import efficientnetb0
from features.bit_resnet import resnet50

extractors = {
        'resnet18': resnet18,
        'resnet18_84': resnet18_84,
        'efficientnetb0' : efficientnetb0, 
        'resnet50': resnet50
        }
