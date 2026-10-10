import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import os
from threading import Lock

app = Flask(__name__)

# Load the model
MODEL_PATH = os.environ.get('CARD_MODEL_PATH',os.environ.get('CARD_MODEL','runs/detect/train5/weights/best.pt'))
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
def generate_frames(camera, release):
    try:
        while True:
            success,frame=camera.read()
            if not success:break
            annotated_frame=model(frame)[0].plot()
            encoded,buffer=cv2.imencode('.jpg',annotated_frame)
            if not encoded:raise RuntimeError('Could not encode camera frame')
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n'
    finally:
        release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if model is None:return jsonify({'error':'Model weights are unavailable'}),503
    if not camera_lock.acquire(blocking=False):return jsonify({'error':'Camera already streaming'}),409
    camera=None
    try:
        camera=cv2.VideoCapture(0)
        if not camera.isOpened():raise RuntimeError('Server camera is unavailable')
    except Exception as error:
        if camera is not None:camera.release()
        camera_lock.release()
        return jsonify({'error':str(error)}),503
    released=False
    def release():
        nonlocal released
        if not released:
            released=True
            try:camera.release()
            finally:camera_lock.release()
    stream=generate_frames(camera,release)
    response=Response(stream,mimetype='multipart/x-mixed-replace; boundary=frame')
    # Closing an unstarted generator does not run its finally clause.
    def cleanup():
        stream.close()
        release()
    response.call_on_close(cleanup)
    return response

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

