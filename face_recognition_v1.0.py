import face_recognition
import cv2
import numpy as np
import time
import pickle
import threading
import queue

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
process_every_n_frames = 2  # Only process every n frames
detection_cooldown = 1.0  # Seconds between repeat announcements for same person

# Set optimal parameters for face recognition
FACE_MODEL = "hog"  # "hog" is faster than "cnn"
FACE_TOLERANCE = 0.6  # Lower is more strict, higher is more permissive

# Shared resources
frame_queue = queue.Queue(maxsize=1)  # Queue for frames to be processed
detection_results = {
    "face_locations": [],
    "face_names": [],
    "timestamp": 0
}
detection_lock = threading.Lock()  # Lock for updating detection results
stop_event = threading.Event()
last_detection_time = {}  # Track when each person was last detected


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
detection_fps = FPSCounter()


def announce_detections(face_names):
    """Announce detected people continuously with cooldown"""
    current_time = time.time()

    # Create a set of unique names (excluding "Unknown")
    unique_names = set([name for name in face_names if name != "Unknown"])

    for name in unique_names:
        # Check if we haven't announced this person recently
        if name not in last_detection_time or (current_time - last_detection_time[name]) >= detection_cooldown:
            print(f"Person detected: {name}")
            last_detection_time[name] = current_time


def process_frames():
    """Process frames for face recognition in a separate thread"""
    frame_counter = 0
    last_detection_time = time.time()

    while not stop_event.is_set():
        try:
            # Get a frame to process
            frame = frame_queue.get(timeout=0.1)
            frame_counter += 1

            # Only process every n frames
            if frame_counter % process_every_n_frames == 0:
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

                    # Announce detections
                    announce_detections(face_names)

                # Scale face locations back to original size
                scaled_locations = [(top * cv_scaler, right * cv_scaler,
                                     bottom * cv_scaler, left * cv_scaler)
                                    for (top, right, bottom, left) in face_locations]

                # Update the shared detection results
                with detection_lock:
                    detection_results["face_locations"] = scaled_locations
                    detection_results["face_names"] = face_names
                    detection_results["timestamp"] = time.time()

                # Calculate detection FPS
                detection_fps.update()

                # Debug info
                process_time = time.time() - detection_start
                if process_time > 0:
                    print(f"Detection processing time: {process_time:.3f}s, FPS: {detection_fps.fps:.1f}")

        except queue.Empty:
            continue

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
    detection_thread = threading.Thread(target=process_frames)
    detection_thread.daemon = True
    detection_thread.start()

    print("Starting facial recognition... Press 'q' to quit.")
    print("Displaying camera feed at max FPS, running detection in background.")

    cv2.namedWindow('Facial Recognition', cv2.WINDOW_NORMAL)

    # Variables for FPS calculation
    last_detection_overlay_time = time.time()
    detection_age = 0
    detection_validity_period = 1.0  # How long to show detection results before considering them stale

    while not stop_event.is_set():
        # Capture frame-by-frame - this happens at maximum speed
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Update the frame queue for processing
        # Replace old frame if queue is full - we only care about the most recent frame
        if not frame_queue.full():
            try:
                frame_queue.put(frame.copy(), block=False)
            except queue.Full:
                pass
        else:
            # If queue is full, replace the old frame
            try:
                _ = frame_queue.get_nowait()
                frame_queue.put(frame.copy(), block=False)
            except (queue.Empty, queue.Full):
                pass

        # Create a display frame
        display_frame = frame.copy()

        # Protect access to detection results with lock
        with detection_lock:
            face_locations = detection_results["face_locations"].copy() if detection_results["face_locations"] else []
            face_names = detection_results["face_names"].copy() if detection_results["face_names"] else []
            detection_timestamp = detection_results["timestamp"]

        # Calculate how old the detection data is
        detection_age = time.time() - detection_timestamp

        # Only overlay detections if they're fresh
        if detection_age < detection_validity_period:
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

        # Update and display FPS
        display_fps.update()

        # Add FPS info to the frame
        cv2.putText(display_frame, f"Display FPS: {display_fps.fps:.1f}", (10, display_frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Detection FPS: {detection_fps.fps:.1f}", (10, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Facial Recognition', display_frame)

        # Check for quit
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()
            break

    # Clean up
    stop_event.set()
    detection_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Facial recognition stopped")


if __name__ == "__main__":
    main()