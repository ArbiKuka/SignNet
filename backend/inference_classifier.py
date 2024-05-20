# Script for opening a camera and checking the hand sign of the user
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
model_dict = pickle.load(open('./main_model.p', 'rb'))
model = model_dict['model']

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands instance
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping of predicted labels to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'HELLO '}

# Variables for delay
delay_frames = 15  # Adjust this value to change delay duration
frame_count = 0
prev_prediction = None
# Main loop for capturing frames and performing hand sign recognition
while True:

    data_aux = []
    x_ = []
    y_ = []

    # Capture frame from the webcam
    ret, frame = cap.read()

    # Get frame dimensions
    H, W, _ = frame.shape
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x and y coordinates of hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize the coordinates and append to data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        try:
            # Predict the hand sign using the trained model
            prediction = model.predict([np.asarray(data_aux)])
            # predicted_character = prediction[9]
            predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw bounding box and predicted hand sign on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 255, 125), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (19, 37, 255), 3,
                        cv2.LINE_AA)

            # Increase frame count
            if prev_prediction == predicted_character:
                frame_count += 1
            else:
                prev_prediction = predicted_character
                frame_count = 0

            # Print prediction after a delay
            if frame_count == delay_frames:
                print(labels_dict[int(prediction[0])], end="")
                frame_count = 0  # Reset frame count

        except ValueError as e:
            # print("Error:", e)  # Print the error message
            pass  # Continue to the next iteration of the loop

    # Display the frame with annotations
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()