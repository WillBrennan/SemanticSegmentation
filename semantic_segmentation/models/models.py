import torch
import torch.nn as nn

from .fcn import FCNResNet101
from .bisenetv2 import BiSeNetV2

models = {
    'FCNResNet101': FCNResNet101,
    'BiSeNetV2': BiSeNetV2,
}


def load_model(model_type, state_dict):
    # todo(will.brennan) - improve this... might want to save a categories file with this instead
    category_prefix = '_categories.'
    categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
    categories = [k[len(category_prefix):] for k in categories]

    model = model_type(categories)
    model.load_state_dict(state_dict)

    return model


class LossWithAux(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_pred, y):
        y_pred_out = y_pred.pop('out')
        loss_output = self.loss_fn(y_pred_out, y)

        loss_aux = [self.loss_fn(y_pred_aux, y) for y_pred_aux in y_pred.values()]
        loss_aux = sum(loss_aux)

        return loss_output + 0.5 * loss_aux
