import io
import json
import base64
import pathlib
import logging
import collections

import cv2
import numpy
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as alb


def _load_image(image_data_b64):
    # note(will.brennan) - from https://github.com/wkentaro/labelme/blob/f20a9425698f1ac9b48b622e0140016e9b73601a/labelme/utils/image.py#L17
    image_data = base64.b64decode(image_data_b64)
    image_data = numpy.fromstring(image_data, dtype=numpy.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class ToTensor(alb.BasicTransform):
    def __init__(self):
        super().__init__(always_apply=True)
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def apply(self, image, **params):
        return self.image_transform(image)

    def apply_to_mask(self, mask, **params):
        return transforms.ToTensor()(mask)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
        }


class LabelMeDataset(data.Dataset):
    def __init__(self, directory: str, use_augmentation: bool, image_height: int = 480, image_width: int = 480):
        self.directory = pathlib.Path(directory)
        self.use_augmentation = use_augmentation
        assert self.directory.exists()
        assert self.directory.is_dir()

        self.labelme_paths = []
        self.categories = collections.defaultdict(list)

        for labelme_path in self.directory.rglob('*.json'):
            with open(labelme_path, 'r') as labelme_file:
                labelme_json = json.load(labelme_file)

                required_keys = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
                assert all(key in labelme_json for key in required_keys), (required_keys, labelme_json.keys())

                self.labelme_paths += [labelme_path]

                for shape in labelme_json['shapes']:
                    label = shape['label']
                    self.categories[label] += [labelme_path]

        for category, paths in self.categories.items():
            for path in paths:
                logging.debug(f'{category} - {path}')
        self.categories = sorted(list(self.categories.keys()))

        logging.info(f'loaded {len(self)} annotations from {self.directory}')
        logging.info(f'use augmentation: {self.use_augmentation}')
        logging.info(f'categories: {self.categories}')

        aug_transforms = [ToTensor()]
        if self.use_augmentation:
            aug_transforms = [
                alb.HueSaturationValue(always_apply=True),
                alb.RandomBrightnessContrast(always_apply=True),
                alb.HorizontalFlip(),
                alb.RandomGamma(always_apply=True),
                alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, always_apply=True),
                alb.PadIfNeeded(min_height=image_height, min_width=image_width, always_apply=True),
                alb.RandomCrop(image_height, image_width, always_apply=True),
            ] + aug_transforms
        else:
            aug_transforms = [
                alb.PadIfNeeded(min_height=image_height, min_width=image_width, always_apply=True),
                alb.CenterCrop(image_height, image_width, always_apply=True),
            ] + aug_transforms

        self.transforms = alb.Compose(transforms=aug_transforms)

    def __len__(self):
        return len(self.labelme_paths)

    def __getitem__(self, idx: int):
        labelme_path = self.labelme_paths[idx]
        logging.debug('parsing labelme json')

        with open(labelme_path, 'r') as labelme_file:
            labelme_json = json.load(labelme_file)

        image_width = labelme_json['imageWidth']
        image_height = labelme_json['imageHeight']

        image = _load_image(labelme_json['imageData'])
        assert image.shape == (image_height, image_width, 3)

        logging.debug('creating segmentation masks per category')

        num_categories = len(self.categories)
        mask = numpy.zeros((num_categories, image_height, image_width), dtype=numpy.float32)

        for shape in labelme_json['shapes']:
            label = self.categories.index(shape['label'])

            points = numpy.array(shape['points']).reshape((-1, 1, 2))
            points = numpy.round(points).astype(numpy.int32)
            cv2.fillPoly(mask[label], [points], (1, ))

        logging.debug('applying transforms to image and mask')
        mask = numpy.transpose(mask, (1, 2, 0))
        target = self.transforms(image=image, mask=mask)

        return target['image'], target['mask']
