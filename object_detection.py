from ultralytics import YOLO
import cv2


def webcam_predict(input_model, label_dict):
    # Turn on webcam for prediction
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame horizontally
        frame = cv2.flip(frame, 1)

        # Display frame with results
        results = input_model.predict(frame, conf=0.6)
        results[0].names = label_dict
        annotated_frame = results[0].plot(conf=False, boxes=True, labels=True)
        cv2.imshow('Webcam YOLOv8', annotated_frame)

        # Exit the loop when ' ' is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


# Translate data.yaml labels to desired labels
card_dict = {
    0: '10 of Diamonds', 1: '10 of Hearts', 2: '10 of Spades', 3: '10 of Clubs',
    4: '2 of Diamonds', 5: '2 of Hearts', 6: '2 of Spades', 7: '2 of Clubs',
    8: '3 of Diamonds', 9: '3 of Hearts', 10: '3 of Spades', 11: '3 of Clubs',
    12: '4 of Diamonds', 13: '4 of Hearts', 14: '4 of Spades', 15: '4 of Clubs',
    16: '5 of Diamonds', 17: '5 of Hearts', 18: '5 of Spades', 19: '5 of Clubs',
    20: '5 of Hearts', 21: '6 of Diamonds', 22: '6 of Hearts', 23: '6 of Spades', 24: '6 of Clubs',
    25: '7 of Diamonds', 26: '7 of Hearts', 27: '7 of Spades', 28: '7 of Clubs',
    29: '8 of Diamonds', 30: '8 of Hearts', 31: '8 of Spades', 32: '8 of Clubs',
    33: '9 of Diamonds', 34: '9 of Hearts', 35: '9 of Spades', 36: '9 of Clubs',
    37: 'Ace of Diamonds', 38: 'Ace of Hearts', 39: 'Ace of Spades', 40: 'Ace of Clubs',
    41: 'Jack of Diamonds', 42: 'Jack of Hearts', 43: 'Jack of Spades', 44: 'Jack of Clubs',
    45: 'King of Diamonds', 46: 'King of Hearts', 47: 'King of Spades', 48: 'King of Clubs',
    49: 'Queen of Diamonds', 50: 'Queen of Hearts', 51: 'Queen of Spades', 52: 'Queen of Clubs'
}


# Load a model
# weights_path = './yolov8n.pt'
weights_path = 'runs/detect/train5/weights/best.pt'


model = YOLO(weights_path)
print("Model successfully loaded")


webcam_predict(model, label_dict=card_dict)
