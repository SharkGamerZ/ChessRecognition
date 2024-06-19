
Chessboard COrners - v5 2024-05-17 6:18pm
==============================

This dataset was exported via roboflow.com on May 17, 2024 at 4:19 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 609 images.
Chess are annotated in YOLOv9 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -45 and +45 degrees
* Random shear of between -17° to +17° horizontally and -18° to +18° vertically
* Random brigthness adjustment of between -11 and +11 percent
* Random Gaussian blur of between 0 and 1.7 pixels
* Salt and pepper noise was applied to 1.76 percent of pixels


