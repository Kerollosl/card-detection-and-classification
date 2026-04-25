# YOLOv8 Playing Card Detection and Classification Model

A YOLOv8 object detection model that identifies and classifies playing cards from a standard 52-card deck in real time. Includes a Flask web app with a live webcam view for quick testing.

## 🚀 Quick Start - Web Interface

The easiest way to test the model is using the web interface:

```bash
# Setup (first time only)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the web app
python web_app.py
```

Then open http://localhost:5003 in your browser!

### Model Architecture
- YOLOv8

### Necessary Packages
- ultralytics: 8.2.76
- opencv-python: 4.8.0.76
- roboflow: 1.1.37
- PyYAML: 6.0.2
- IPython: 7.34.0
- flask: 3.0.0

## 📋 Usage Options

### Option 1: Web Interface (Recommended)

1. **Setup environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the web app:**
   ```bash
   python web_app.py
   ```

3. **Access the interface:**
   - Open http://localhost:5003 in your browser
   - Allow camera permissions when prompted
   - Hold a playing card in front of your webcam
   - The model will detect and classify the card in real-time

### Option 2: Command Line (Original Method)

1. In a Python notebook environment, upload the `/runs/detect/train5/weights/best.pt` file to allow the model to begin training with pre-trained weights. Run the `train_and_validate.ipynb` notebook. Note: This process is done in a notebook to leverage simplified downloading of the Roboflow dataset and use of a virtual GPU.

2. Once the notebook has been run, download the created `runs.zip` file. Unzip and upload to the same directory as the `test.py` script. Navigate in the runs directory to find the path for the new `best.pt` file. Change the path in the `test.py` script on line 54 to match this path.

3. Run the `test.py` script:
   ```bash
   python test.py
   ```
   This will open a webcam for testing the newly trained model. In the webcam display, show any standard playing card from a 52-card deck (does not work with Joker cards) and the model will attempt to detect the card and label which card it is.

## 📁 Contents

- `web_app.py` - Flask web application with modern UI for real-time card detection
- `templates/index.html` - Web interface template with TailwindCSS
- `test.py` - Command-line program to load model and test it with live webcam
- `train_and_validate.ipynb` - Program to download Roboflow dataset, load, train, and validate YOLOv8 model
- `requirements.txt` - Python dependencies
- `/runs/detect` - Subdirectory of YOLOv8 model training and validation containing the training weights and validation metrics
- `/runs/detect/train5/weights/best.pt` - Highest performing model weights file from previous trainings

## 🎯 Features

- ✅ Real-time card detection and classification
- ✅ Modern web interface with live video feed
- ✅ Support for all 52 standard playing cards
- ✅ Visual bounding boxes and labels
- ✅ Confidence scores
- ✅ Health monitoring endpoint

## 🔧 Technical Details

- **Framework:** Ultralytics YOLOv8
- **Backend:** Flask + OpenCV
- **Frontend:** HTML + TailwindCSS
- **Model:** Custom trained on playing card dataset
- **Input:** Webcam (real-time)
- **Output:** Bounding boxes with card labels

## 📝 Notes

- Works with standard 52-card deck only (Jokers not supported)
- Ensure good lighting for best detection results
- Camera permissions must be granted for web interface
- Model weights must be present at `runs/detect/train5/weights/best.pt`