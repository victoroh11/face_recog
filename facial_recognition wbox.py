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
process_every_n_frames = 3  # Only process every n frames
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
frame_counter = 0
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
last_detection_time = {}  # Track when each person was last detected
detection_cooldown = 1.0  # Seconds between repeat announcements for same person

# Set optimal parameters for face recognition
FACE_MODEL = "hog"  # "hog" is faster than "cnn"
FACE_TOLERANCE = 0.6  # Lower is more strict, higher is more permissive


def capture_frames(cap, frame_queue, stop_event):
    """Continuously capture frames in a separate thread"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            stop_event.set()
            break

        # Skip frames if processing can't keep up
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Discard oldest frame
            except queue.Empty:
                pass

        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            pass

        time.sleep(0.01)  # Small sleep to prevent maxing out CPU


def announce_detections(face_names):
    """Announce detected people continuously with cooldown"""
    global last_detection_time
    current_time = time.time()

    # Create a set of unique names (excluding "Unknown")
    unique_names = set([name for name in face_names if name != "Unknown"])

    for name in unique_names:
        # Check if we haven't announced this person recently
        if name not in last_detection_time or (current_time - last_detection_time[name]) >= detection_cooldown:
            print(f"Person detected: {name}")
            last_detection_time[name] = current_time


def process_frames(frame_queue, result_queue, stop_event):
    """Process frames for face recognition in a separate thread"""
    global face_locations, face_names, frame_counter

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            frame_counter += 1

            # Only process every n frames
            if frame_counter % process_every_n_frames == 0:
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Face detection
                face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_MODEL)

                # Only compute encodings if faces detected
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []

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

                    # Continuously announce detections
                    announce_detections(face_names)
                else:
                    face_names = []

                # Store results
                result = {
                    "frame": frame,
                    "face_locations": [(top * cv_scaler, right * cv_scaler,
                                        bottom * cv_scaler, left * cv_scaler)
                                       for (top, right, bottom, left) in face_locations],
                    "face_names": face_names
                }

                # Discard old result if process can't keep up
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    result_queue.put(result, block=False)
                except queue.Full:
                    pass

        except queue.Empty:
            continue


def calculate_fps():
    """Calculate frames per second"""
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


def draw_results(frame, face_locations, face_names):
    """Draw face boxes and names on the frame"""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2)

        # Draw label with name
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    return frame


# Main function
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

    # Start threads
    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event))
    capture_thread.daemon = True
    capture_thread.start()

    process_thread = threading.Thread(target=process_frames, args=(frame_queue, result_queue, stop_event))
    process_thread.daemon = True
    process_thread.start()

    print("Starting facial recognition... Press 'q' to quit.")
    print("Detected people will be continuously announced in the console.")

    last_result = None

    # Main loop - only handles display
    while not stop_event.is_set():
        # Try to get latest result
        try:
            result = result_queue.get(block=False)
            last_result = result
        except queue.Empty:
            pass

        # If we have a result, display it
        if last_result is not None:
            display_frame = last_result["frame"].copy()
            display_frame = draw_results(display_frame, last_result["face_locations"], last_result["face_names"])

            # Also add text showing currently detected people
            if last_result["face_names"]:
                # Create a string of unique names (remove duplicates)
                unique_names = set([name for name in last_result["face_names"] if name != "Unknown"])
                if unique_names:
                    names_text = "Detected: " + ", ".join(unique_names)
                    cv2.putText(display_frame, names_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate and display FPS
            current_fps = calculate_fps()
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Facial Recognition', display_frame)

        # Check for quit
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()
            break

    # Clean up
    stop_event.set()
    capture_thread.join(timeout=1.0)
    process_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Facial recognition stopped")


if __name__ == "__main__":
    main()