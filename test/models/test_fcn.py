from semantic_segmentation.models import FCNResNet101

import torch


def test_FCNResNet101():
    model = FCNResNet101(['skin', 'faces'])

    image = torch.zeros((4, 3, 128, 256))
    output = model(image)

    seg_mask = output['out']
    aux_mask = output['aux']

    assert seg_mask.shape == torch.Size([4, 2, 128, 256])
    assert seg_mask.dtype == torch.float32

    assert aux_mask.shape == torch.Size([4, 2, 128, 256])
    assert aux_mask.dtype == torch.float32


def test_FCNResNet101_eval():
    model = FCNResNet101(['skin', 'faces'])

    with torch.no_grad():
        model.eval()
        image = torch.zeros((4, 3, 128, 256))
        output = model(image)

        seg_mask = output['out']
        aux_mask = output['aux']

        assert seg_mask.shape == torch.Size([4, 2, 128, 256])
        assert seg_mask.dtype == torch.float32

        assert aux_mask.shape == torch.Size([4, 2, 128, 256])
        assert aux_mask.dtype == torch.float32
