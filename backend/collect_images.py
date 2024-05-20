# Script for collecting images of the user
import os
import cv2

# Directory to store collected data
DATA_DIR = './data1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (hand signs) and size of the dataset per class
number_of_classes = 25
dataset_size = 100

# Open video capture device (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for the current class
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Display a message to prompt the user to prepare for image capture
    done = False
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Couldn't read frame. Exiting...")
            break

        cv2.putText(frame, 'Press Q when ready.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
