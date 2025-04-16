import face_recognition
import cv2
import numpy as np
import time
import pickle
import threading

# Load pre-trained face encodings
print("[INFO] loading encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    print(f"Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} unique people")
except Exception as e:
    print(f"Error loading encodings: {e}")
    exit()

# Initialize variables
cv_scaler = 4  # Balance between speed and accuracy
detection_results = {
    "face_locations": [],
    "face_names": [],
    "timestamp": 0,
    "processing": False
}
detection_lock = threading.Lock()  # Lock for updating detection results
stop_event = threading.Event()
detect_event = threading.Event()  # Event to trigger face detection

# Set optimal parameters for face recognition
FACE_MODEL = "hog"  # "hog" is faster than "cnn"
FACE_TOLERANCE = 0.6  # Lower is more strict, higher is more permissive


# FPS calculation
class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps


display_fps = FPSCounter()


def detect_faces(frame):
    """Process a single frame for face detection and return results"""
    # Set processing flag
    with detection_lock:
        detection_results["processing"] = True

    detection_start = time.time()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_MODEL)
    face_names = []

    # Only compute encodings if faces detected
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,
                                                     tolerance=FACE_TOLERANCE)
            name = "Unknown"

            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)

    # Scale face locations back to original size
    scaled_locations = [(top * cv_scaler, right * cv_scaler,
                         bottom * cv_scaler, left * cv_scaler)
                        for (top, right, bottom, left) in face_locations]

    # Calculate processing time
    process_time = time.time() - detection_start

    # Update the shared detection results
    with detection_lock:
        detection_results["face_locations"] = scaled_locations
        detection_results["face_names"] = face_names
        detection_results["timestamp"] = time.time()
        detection_results["processing"] = False

    # Print results
    print(f"Detection completed in {process_time:.3f}s")
    if face_names:
        unique_names = set([name for name in face_names if name != "Unknown"])
        if unique_names:
            print(f"Detected: {', '.join(unique_names)}")
    else:
        print("No faces detected")

    return scaled_locations, face_names


def detection_worker():
    """Background worker thread for face detection"""
    while not stop_event.is_set():
        # Wait for detection request
        detect_event.wait(timeout=0.1)

        if detect_event.is_set() and not stop_event.is_set():
            # Clear the event
            detect_event.clear()

            # Get the current frame
            with detection_lock:
                if hasattr(detection_worker, 'current_frame') and detection_worker.current_frame is not None:
                    frame = detection_worker.current_frame.copy()
                else:
                    continue

            # Process the frame
            detect_faces(frame)

    print("Detection thread stopped")


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Use a lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set buffer size to minimum to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Start the detection thread
    detection_thread = threading.Thread(target=detection_worker)
    detection_thread.daemon = True
    detection_thread.start()

    print("Starting webcam... Press SPACE to detect faces, 'q' to quit.")

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

    # Variable to track if we should show detection results
    show_detection = False
    detection_display_duration = 3.0  # Show detection results for 3 seconds
    detection_display_until = 0

    while not stop_event.is_set():
        # Capture frame-by-frame - this happens at maximum speed
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Store the current frame for the detection thread
        detection_worker.current_frame = frame.copy()

        # Create a display frame
        display_frame = frame.copy()

        # Check if we should show detection results
        current_time = time.time()
        show_detection = current_time < detection_display_until

        # If we're showing detection and not currently processing
        if show_detection:
            # Protect access to detection results with lock
            with detection_lock:
                face_locations = detection_results["face_locations"]
                face_names = detection_results["face_names"]
                processing = detection_results["processing"]

            if not processing:
                # Draw boxes and labels for each face
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Draw box around face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (244, 42, 3), 2)

                    # Draw label with name
                    cv2.rectangle(display_frame, (left, bottom - 25), (right, bottom), (244, 42, 3), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(display_frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

                # Add text showing currently detected people
                if face_names:
                    unique_names = set([name for name in face_names if name != "Unknown"])
                    if unique_names:
                        names_text = "Detected: " + ", ".join(unique_names)
                        cv2.putText(display_frame, names_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check if we're currently processing a detection
        with detection_lock:
            processing = detection_results["processing"]

        # Show processing status
        if processing:
            cv2.putText(display_frame, "Processing...", (display_frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update and display FPS
        display_fps.update()

        # Add FPS info to the frame
        cv2.putText(display_frame, f"FPS: {display_fps.fps:.1f}", (10, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add instructions
        cv2.putText(display_frame, "Press SPACE to detect faces", (10, display_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Webcam', display_frame)

        # Check for key presses
        key = cv2.waitKey(1)

        # Check for quit
        if key == ord('q'):
            stop_event.set()
            break

        # Check for space bar (trigger detection)
        elif key == 32:  # Space bar
            with detection_lock:
                processing = detection_results["processing"]

            if not processing:
                print("Face detection triggered")
                # Set the event to trigger detection
                detect_event.set()
                # Set the time to display results
                detection_display_until = time.time() + detection_display_duration
            else:
                print("Detection already in progress")

    # Clean up
    stop_event.set()
    detect_event.set()  # Set event to ensure detection thread exits its wait
    detection_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Program stopped")


if __name__ == "__main__":
    main()