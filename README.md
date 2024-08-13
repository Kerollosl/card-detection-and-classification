# YOLOv8 Playing Card Detection and Classification Model

## Kerollos Lowandy

**Repository: card-detection-and-classification**

## GitHub Link
[https://github.com/Kerollosl/card-detection-and-classification](https://github.com/Kerollosl/card-detection-and-classification)

### Model Architecture
- YOLOv8

### Necessary Packages
- ultralytics: 8.2.76
- cv2: 4.8.0.76
- roboflow: 1.1.37
- yaml: 6.0.2
- IPython: 7.34.0

### Directions

1. In a Python notebook environment, upload the `/runs/detect/train5/weights/best.pt` file to allow the model to begin training with pre-trained weights. Run the `train_and_validate.ipynb` notebook. Note: This process is done in a notebook to leverage simplified downloading of the Roboflow dataset and use of a virtual GPU.
2. Once the notebook has been run, download the created `runs.zip` file. Unzip and upload to the same directory as the `test.py` script. Navigate in the runs directory to find the path for the new `best.pt` file. Change the path in the `test.py` script on line 54 to match this path. 
3. Run the `test.py` script. This will open a webcam for testing the newly trained model. In the webcam display, show any standard playing card from a 52-card deck (does not work with Joker cards) and the model will attempt to detect the card and label which card it is.


### Contents

- `test.py` - Program to load model and test it with live webcam
- `train_and_validate.ipynb` - Program to download Roboflow dataset, load, train, and validate YOLOv8 model, and save weights and validation metrics to a zip file to be downloaded. 
- `/runs/detect` - Subdirectory of YOLOv8 model training and validation containing the training weights and validation metrics.
- `/runs/detect/train5/weights/best.pt` - Highest performing model weights file from previous trainings