import cv2
import os
from datetime import datetime
import time

# Change this to the name of the person you're photographing
PERSON_NAME = "nicholas"


def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder


def capture_photos(name):
    folder = create_folder(name)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set resolution - adjust these values based on your Logitech camera's capabilities
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow camera to warm up
    time.sleep(2)

    photo_count = 0

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the frame
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):  # Q key
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")


if __name__ == "__main__":
    capture_photos(PERSON_NAME)