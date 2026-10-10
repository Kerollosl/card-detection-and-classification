import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import os
from threading import Lock

app = Flask(__name__)

# Load the model
MODEL_PATH = os.environ.get('CARD_MODEL','runs/detect/train5/weights/best.pt')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Please ensure the model weights are in the correct location.")

load_model()

camera_lock=Lock()
def generate_frames():
    # A physical camera supports one active stream; report conflicts explicitly.
    if not camera_lock.acquire(blocking=False):
        return
    camera=None
    try:
        camera=cv2.VideoCapture(0)
        if not camera.isOpened():raise RuntimeError('Server camera is unavailable')
        while True:
            success,frame=camera.read()
            if not success:break
            annotated_frame=model(frame)[0].plot() if model is not None else frame
            encoded,buffer=cv2.imencode('.jpg',annotated_frame)
            if not encoded:raise RuntimeError('Could not encode camera frame')
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n'
    finally:
        if camera is not None:camera.release()
        camera_lock.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if model is None:return jsonify({'error':'Model weights are unavailable'}),503
    if camera_lock.locked():return jsonify({'error':'Camera already streaming'}),409
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    print("🚀 Starting Card Detection Web App")
    print("📍 Open http://localhost:5003 in your browser")
    app.run(host=os.environ.get('CARD_HOST', '127.0.0.1'), port=int(os.environ.get('CARD_PORT', '5003')), debug=os.environ.get('CARD_DEBUG') == '1')

