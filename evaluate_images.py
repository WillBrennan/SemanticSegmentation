import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision import transforms

from semantic_segmentation import FCNResNet101
from semantic_segmentation import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    assert args.display or args.save

    logging.info(f'loading model from {args.model}')
    model = FCNResNet101.load(torch.load(args.model))
    model.cuda().eval()

    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: cv2.imread(str(image_path))),
            transforms.Lambda(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        logging.info(f'segmenting {image_file} with threshold of {args.threshold}')

        image = fn_image_transform(image_file)

        with torch.no_grad():
            image = image.cuda().unsqueeze(0)
            results = model(image)['out']
            results = torch.sigmoid(results)

            results = results > args.threshold

        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                output_name = f'results_{category}_{image_file.name}'
                logging.info(f'writing output to {output_name}')
                cv2.imwrite(str(output_name), category_image)
                cv2.imwrite(f'mask_{category}_{image_file.name}', mask_image)

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f'mask_{category}', mask_image)

        if args.display:
            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()
