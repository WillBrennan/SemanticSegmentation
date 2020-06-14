import logging

import torch
from torch import nn
from torchvision import models


class FCNResNet101(nn.Module):
    @staticmethod
    def load(state_dict):
        # todo(will.brennan) - improve this... might want to save a categories file with this instead
        category_prefix = '_categories.'
        categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
        categories = [k[len(category_prefix):] for k in categories]

        model = FCNResNet101(categories)
        model.load_state_dict(state_dict)
        return model

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


class LossWithAux(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_pred, y):
        loss_output = self.loss_fn(y_pred['out'], y)
        loss_aux = self.loss_fn(y_pred['aux'], y)

        return loss_output + 0.5 * loss_aux
