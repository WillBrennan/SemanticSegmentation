# Semantic Segmentation
## Overview
This project started as a replacement to the [Skin Detection](https://github.com/WillBrennan/SkinDetector) project that used traditional computer vision techniques. This project fine-tunes `fcn_resnet101_coco` from torchvision on masks annotated using labelme. As labelme annotations allow for multiple categories per a pixel we use multi-label semantic segmentation. 

Currently this project is optimized for accuracy over being real-time, if people want a real-time segmentation network then make an issue letting me know!

## Getting Started
This project uses conda to manage its enviroment; once conda is installed we create the enviroment and activate it, 
```bash
conda env create -f enviroment.yml
conda activate semantic_segmentation
```
. On windows powershell needs to be initialised and the execution policy needs to be modified. 
```bash
conda init powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Pre-Trained Segmentation Projects
This project comes bundled with several pretrained models, which can be found in the `pretrained` directory. To infer segmentation masks on your images run `evaluate_images`.
```bash
# to display the output
python evaluate_images.py --images ~/Pictures/ --model pretrained/model_segmentation_skin_30.pth --display
# to save the output
python evaluate_images.py --images ~/Pictures/ --model pretrained/model_segmentation_skin_30.pth --save
```
### Skin Segmentation
This model was trained with a custom dataset of 150 images taken from COCO where skin segmentation annotations were added. This includes a wide variety of skin colours and lighting conditions making it more robust than the [Skin Detection](https://github.com/WillBrennan/SkinDetector) project. This model detects, 

- skin
![Skin Segmentation](https://raw.githubusercontent.com/WillBrennan/SemanticSegmentation/master/pretrained/skin_examples.png)

### Pizza Topping Segmentation
This was trained with a custom dataset of 89 images taken from COCO where pizza topping annotations were added. There's very few images for each type of topping so this model performs very badly and needs quite a few more images to behave well!

- 'chilli', 'ham', 'jalapenos', 'mozzarella', 'mushrooms', 'olive', 'pepperoni', 'pineapple', 'salad', 'tomato'

![Pizza Toppings](https://raw.githubusercontent.com/WillBrennan/SemanticSegmentation/master/pretrained/pizza_toppings_example.png)

### Cat and Bird Segmentation
Annotated images of birds and cats were taken from COCO using the `extract_from_coco` script and then trained on. 

- cat, birds

![Demo on Cat & Birds](https://raw.githubusercontent.com/WillBrennan/SemanticSegmentation/master/pretrained/cat_examples.png)


## Training New Projects
To train a new project you can either create new labelme annotations on your images, to launch labelme run, 

```bash
labelme
```
and start annotating your images! You'll need a couple of hundred. Alternatively if your category is already in COCO you can run the conversion tool to create labelme annotations from them. 

```bash
python extract_from_coco.py --images ~/datasets/coco/val2017 --annotations ~/datasets/coco/annotations/instances_val2017.json --output ~/datasets/my_cat_images_val --categories cat
```

Once you've got a directory of labelme annotations you can check how the images will be shown to the model during training by running, 

```bash
python check_dataset.py --dataset ~/datasets/my_cat_images_val
# to show our dataset with training augmentation
python check_dataset.py --dataset ~/datasets/my_cat_images_val --use-augmentation
```
. If your happy with the images and how they'll appear in training then train the model using, 

```bash
python train.py --train ~/datasets/my_cat_images_train --val ~/datasets/my_cat_images_val --model-tag segmentation_cat
```
. This may take some time depending on how many images you have. Tensorboard logs are available in the `logs` directory. To run your trained model on a directory of images run

```bash
# to display the output
python evaluate_images.py --images ~/Pictures/my_cat_imgs --model models/model_segmentation_cat_30.pth --display
# to save the output
python evaluate_images.py --images ~/Pictures/my_cat_imgs --model models/model_segmentation_cat_30.pth --save
```
