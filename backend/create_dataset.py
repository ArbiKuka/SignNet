import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Function to adjust brightness of an image
def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Initialize Mediapipe for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hands object for detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the data
DATA_DIR = './data'

# Initialize lists to store data and corresponding labels
data = []
labels = []

# Maximum number of landmarks (adjust according to your needs)
max_landmarks = 21

# Brightness adjustment factors
brightness_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Adjust as needed

# Iterate through each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip if not a directory
        continue
    # Iterate through each image in the directory
    for img_path in os.listdir(dir_path):
        # Read the original image
        img = cv2.imread(os.path.join(dir_path, img_path))

        # Apply brightness augmentation multiple times
        for brightness_factor in brightness_factors:
            # Adjust brightness
            brightened_img = adjust_brightness(img, brightness_factor)

            # Process the brightened image to detect hand landmarks
            results = hands.process(brightened_img)
            if results.multi_hand_landmarks:
                # Initialize auxiliary list to store normalized landmark data
                data_aux = []
                # Loop through each detected hand landmark
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract x and y coordinates of each landmark
                    x_ = [lmk.x for lmk in hand_landmarks.landmark]
                    y_ = [lmk.y for lmk in hand_landmarks.landmark]

                    # Normalize the coordinates based on the minimum x and y values
                    min_x, min_y = min(x_), min(y_)
                    for x, y in zip(x_, y_):
                        data_aux.append(x - min_x)
                        data_aux.append(y - min_y)

                # Pad or truncate data_aux to a fixed length
                if len(data_aux) < max_landmarks * 2:
                    data_aux.extend([0.0] * (max_landmarks * 2 - len(data_aux)))
                elif len(data_aux) > max_landmarks * 2:
                    data_aux = data_aux[:max_landmarks * 2]

                # Append normalized data and corresponding label to lists
                data.append(data_aux)
                labels.append(dir_)

# Serialize the data and labels into a pickle file
with open('data_with_brightness.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# import os  # Operating System module for directory operations
# import pickle  # Module for serializing and deserializing Python objects
# import mediapipe as mp  # Mediapipe library for hand detection and landmark estimation
# import cv2  # OpenCV library for image processing
#
# # Initialize Mediapipe for hand landmark detection
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Initialize hands object for detection
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# # Directory containing the data
# DATA_DIR = './data'
#
# # Initialize lists to store data and corresponding labels
# data = []
# labels = []
#
# # Maximum number of landmarks (adjust according to your needs)
# max_landmarks = 21
#
# # Iterate through each directory in the data directory
# for dir_ in os.listdir(DATA_DIR):
#     dir_path = os.path.join(DATA_DIR, dir_)
#     if not os.path.isdir(dir_path):  # Skip if not a directory
#         continue
#     # Iterate through each image in the directory
#     for img_path in os.listdir(dir_path):
#         data_aux = []  # Auxiliary list to store normalized landmark data
#
#         x_ = []  # List to store x coordinates
#         y_ = []  # List to store y coordinates
#
#         # Read the image
#         img = cv2.imread(os.path.join(dir_path, img_path))
#         # Convert image to RGB format
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # Process the image to detect hand landmarks
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             # Loop through each detected hand landmark
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Extract x and y coordinates of each landmark
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#
#                     x_.append(x)  # Append x coordinate
#                     y_.append(y)  # Append y coordinate
#
#                 # Normalize the coordinates based on the minimum x and y values
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))  # Normalize x coordinate
#                     data_aux.append(y - min(y_))  # Normalize y coordinate
#
#             # Pad or truncate data_aux to a fixed length
#             if len(data_aux) < max_landmarks * 2:
#                 data_aux.extend([26.26] * (max_landmarks * 2 - len(data_aux)))
#             elif len(data_aux) > max_landmarks * 2:
#                 data_aux = data_aux[:max_landmarks * 2]
#
#             # Append normalized data and corresponding label to lists
#             data.append(data_aux)
#             labels.append(dir_)
#
# # Serialize the data and labels into a pickle file
# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()