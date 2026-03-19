"""
Card Detection and Classification using YOLOv8
This script performs real-time card detection using a webcam.
"""

from ultralytics import YOLO
import cv2
import time
import argparse

def run_webcam_prediction(model, label_dict, conf_threshold=0.6, camera_index=0):
    """
    Run real-time card detection using the webcam.

    Args:
        model: Trained YOLO model for card detection.
        label_dict: Dictionary mapping class indices to card names.
        conf_threshold: Minimum confidence threshold for detections.
        camera_index: The system index for the webcam device.
    """
    # Initialize webcam capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}. Please ensure a webcam is connected and available.")
        return

    # Set resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Starting webcam prediction. Press 'q' to quit.")

    try:
        prev_time = time.time()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            # Mirror frame horizontally for natural viewing
            frame = cv2.flip(frame, 1)

            # Perform prediction
            results = model.predict(frame, conf=conf_threshold, verbose=False)

            # Override the labels in the results object directly so .plot() uses them
            # Ensure results[0].names exists before updating
            if results and results[0] and hasattr(results[0], 'names'):
                results[0].names.update(label_dict)

            # Annotate frame with detections
            annotated_frame = results[0].plot(conf=False, boxes=True, labels=True)

            # --- Cool UI Additions: HUD Overlay ---
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Create a semi-transparent background for HUD
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

            # Add HUD text (FPS and Card Count)
            num_cards = len(results[0].boxes)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Cards Detected: {num_cards}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # --------------------------------------

            # Display the annotated frame
            cv2.imshow('Card Detection - YOLOv8', annotated_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam prediction stopped.")


# Mapping of class indices to card names
CARD_LABELS = {
    0: '10 of Diamonds', 1: '10 of Hearts', 2: '10 of Spades', 3: '10 of Clubs',
    4: '2 of Diamonds', 5: '2 of Hearts', 6: '2 of Spades', 7: '2 of Clubs',
    8: '3 of Diamonds', 9: '3 of Hearts', 10: '3 of Spades', 11: '3 of Clubs',
    12: '4 of Diamonds', 13: '4 of Hearts', 14: '4 of Spades', 15: '4 of Clubs',
    16: '5 of Diamonds', 17: '5 of Hearts', 18: '5 of Spades', 19: '5 of Clubs',
    20: '6 of Diamonds', 21: '6 of Hearts', 22: '6 of Spades', 23: '6 of Clubs',
    24: '7 of Diamonds', 25: '7 of Hearts', 26: '7 of Spades', 27: '7 of Clubs',
    28: '8 of Diamonds', 29: '8 of Hearts', 30: '8 of Spades', 31: '8 of Clubs',
    32: '9 of Diamonds', 33: '9 of Hearts', 34: '9 of Spades', 35: '9 of Clubs',
    36: 'Ace of Diamonds', 37: 'Ace of Hearts', 38: 'Ace of Spades', 39: 'Ace of Clubs',
    40: 'Jack of Diamonds', 41: 'Jack of Hearts', 42: 'Jack of Spades', 43: 'Jack of Clubs',
    44: 'King of Diamonds', 45: 'King of Hearts', 46: 'King of Spades', 47: 'King of Clubs',
    48: 'Queen of Diamonds', 49: 'Queen of Hearts', 50: 'Queen of Spades', 51: 'Queen of Clubs'
}


if __name__ == '__main__':
    # Set up argument parsing to avoid hardcoding variables
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 Card Detection")
    parser.add_argument('--weights', type=str, default='runs/detect/train5/weights/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--source', type=int, default=0, help='Webcam index')
    args = parser.parse_args()

    try:
        # Load the YOLO model
        model = YOLO(args.weights)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.weights}'. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Run webcam prediction
    run_webcam_prediction(model, CARD_LABELS, conf_threshold=args.conf, camera_index=args.source)
