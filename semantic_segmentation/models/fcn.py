import logging

import torch
from torch import nn
from torchvision import models


class FCNResNet101(nn.Module):
    def __init__(self, categories):
        super().__init__()
        logging.info(f'creating model with categories: {categories}')

        # todo(will.brennan) - find a nicer way of saving the categories in the state dict...
        self._categories = nn.ParameterDict({i: nn.Parameter(torch.Tensor(0)) for i in categories})
        num_categories = len(self._categories)

        self.model = models.segmentation.fcn_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_categories, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_categories, 1)

    @property
    def categories(self):
        return self._categories

    def forward(self, image: torch.Tensor):
        return self.model(image)