from typing import List
from typing import Dict
import itertools

import torch
import cv2
import numpy


def draw_results(
    image: torch.Tensor,
    mask: torch.Tensor,
    categories: List[str],
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225)
):
    assert mask.shape[0] == len(categories)
    assert image.shape[1:] == mask.shape[1:]
    assert mask.dtype == torch.bool

    image = image.cpu().numpy()
    image = numpy.transpose(image, (1, 2, 0))
    image = (image * img_std) + img_mean
    image = (255 * image).astype(numpy.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = mask.cpu().numpy()

    colours = (
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    )

    for label, (category, category_mask) in enumerate(zip(categories, mask)):
        cat_image = image.copy()

        cat_colour = colours[label % len(colours)]
        cat_colour = numpy.array(cat_colour)
        cat_image[category_mask] = 0.5 * cat_image[category_mask] + 0.5 * cat_colour

        mask_image = image.copy()
        mask_image[~category_mask] = 0

        yield category, cat_image, mask_image
