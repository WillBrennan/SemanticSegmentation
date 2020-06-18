import argparse
import logging
import pathlib
import json
import collections
import multiprocessing as mp
import functools
import shutil
import base64

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    parser.add_argument('--categories', type=str, nargs='+', required=True)

    parser.add_argument('--num-workers', type=int, default=None)

    return parser.parse_args()


def images_with_categories(annotations, categories):
    all_categories = {i['name']: i['id'] for i in annotations['categories']}
    category_names = {i['id']: i['name'] for i in annotations['categories']}
    categories = [i for i in categories if i in all_categories]
    category_ids = [all_categories[i] for i in categories]

    logging.info(f'categories: {categories}')
    logging.info(f'all available categories: {all_categories.keys()}')

    anns_by_image = collections.defaultdict(list)

    for ann in annotations['annotations']:
        category_id = ann['category_id']
        image_id = ann['image_id']

        if category_id not in category_ids:
            continue

        ann['category'] = category_names[category_id]
        anns_by_image[image_id].append(ann)

    for image_info in annotations['images']:
        image_id = image_info['id']
        if image_id in anns_by_image:
            anns_for_image = anns_by_image[image_id]
            yield image_info, anns_for_image


def ann_to_shape(ann):
    points = ann['segmentation'][0]
    points = [[x, y] for x, y in zip(points[0::2], points[1::2])]

    return {
        'label': ann['category'],
        'points': points,
        'group_id': None,
        'shape_type': 'polygon',
        'flags': {},
    }


def save_labelme(image_info, anns, images_dir: pathlib.Path, output_dir: pathlib.Path):
    image_path = images_dir / image_info['file_name']
    output_image_path = output_dir / image_info['file_name']

    logging.info(f'reading image from {image_path}')

    assert image_path.exists()
    shutil.copyfile(image_path, output_image_path)

    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode()

    # warning(will.brennan):
    # currently we're only handling 'segmentation' being points... 
    # maybe use pycocotools despite it not working on windows...
    anns = [ann for ann in anns if len(ann['segmentation']) >= 1 and isinstance(ann['segmentation'], list)]
    shapes = [ann_to_shape(ann) for ann in anns]

    labelme_data = {
        'version': '4.2.10',
        'flags': {},
        'shapes': shapes,
        'imagePath': image_info['file_name'],
        'imageHeight': image_info['height'],
        'imageWidth': image_info['width'],
        'imageData': image_data,
    }

    output_json_path = output_image_path.with_suffix('.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(labelme_data, json_file)


if __name__ == '__main__':
    # note(will.brennan) - not using pycocotools because the authors refuse to support windows
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info(f'using annotations from {args.annotations}')
    with open(args.annotations, 'r') as annotations_file:
        annotations = json.load(annotations_file)

    gn_anns = images_with_categories(annotations, args.categories)

    num_workers = mp.cpu_count() if args.num_workers is None else args.num_workers
    images_dir = pathlib.Path(args.images)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f'saving labelme data to {output_dir} with {num_workers}')

    fn_save_labelme = functools.partial(save_labelme, images_dir=images_dir, output_dir=output_dir)

    pool = mp.Pool(num_workers)
    res = pool.starmap_async(fn_save_labelme, gn_anns)
    res.get()
