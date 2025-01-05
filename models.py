import torch
from torch import nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetClassifier, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*(list(self.resnet.children())[:-1]))

        num_features = self.resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits

