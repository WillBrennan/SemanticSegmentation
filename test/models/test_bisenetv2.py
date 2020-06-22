from semantic_segmentation.models import BiSeNetV2

import torch


def test_BiSeNetV2101():
    model = BiSeNetV2(['skin', 'faces'])

    image = torch.zeros((4, 3, 128, 256))
    output = model(image)

    assert list(output.keys()) == ['out', 'aux_c2', 'aux_c3', 'aux_c4', 'aux_c5']

    for mask in output.values():
        assert mask.shape == torch.Size([4, 2, 128, 256])
        assert mask.dtype == torch.float32


def test_BiSeNetV2_eval():
    model = BiSeNetV2(['skin', 'faces'])

    with torch.no_grad():
        model.eval()
        image = torch.zeros((4, 3, 128, 256))
        output = model(image)

    assert list(output.keys()) == ['out', 'aux_c2', 'aux_c3', 'aux_c4', 'aux_c5']

    for mask in output.values():
        assert mask.shape == torch.Size([4, 2, 128, 256])
        assert mask.dtype == torch.float32


def test_BiSeNetV2_pizza():
    categories = [
        'chilli', 'ham', 'jalapenos', 'mozzarella', 'mushrooms', 'olive', 'pepperoni', 'pineapple', 'salad', 'tomato'
    ]
    model = BiSeNetV2(categories)

    image = torch.zeros((4, 3, 256, 512))
    output = model(image)

    assert list(output.keys()) == ['out', 'aux_c2', 'aux_c3', 'aux_c4', 'aux_c5']

    for mask in output.values():
        assert mask.shape == torch.Size([4, 10, 256, 512])
        assert mask.dtype == torch.float32


def test_BiSeNetV2_pizza_eval():
    categories = [
        'chilli', 'ham', 'jalapenos', 'mozzarella', 'mushrooms', 'olive', 'pepperoni', 'pineapple', 'salad', 'tomato'
    ]
    model = BiSeNetV2(categories)

    with torch.no_grad():
        model.eval()
        image = torch.zeros((4, 3, 128, 256))
        output = model(image)

    assert list(output.keys()) == ['out', 'aux_c2', 'aux_c3', 'aux_c4', 'aux_c5']

    for mask in output.values():
        assert mask.shape == torch.Size([4, 10, 128, 256])
        assert mask.dtype == torch.float32
