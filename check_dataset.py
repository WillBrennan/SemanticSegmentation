import argparse
import logging

import cv2

from semantic_segmentation import LabelMeDataset
from semantic_segmentation import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--use-augmentation', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def log_stats(name, data):
    data_type = data.dtype
    data = data.float()
    logging.info(f'{name} - {data.shape} - {data_type} - min: {data.min()} mean: {data.mean()} max: {data.max()}')


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    dataset = LabelMeDataset(args.dataset, args.use_augmentation)

    num_samples = len(dataset)
    for idx in range(num_samples):
        logging.info(f'showing {(idx + 1)} of {num_samples} samples')
        image, mask = dataset[idx]

        log_stats('image', image)
        log_stats('mask', mask)

        mask = mask > 0.5

        for category, category_image in draw_results(image, mask, categories=dataset.categories):
            cv2.imshow(category, category_image)

        if cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()
