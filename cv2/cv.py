import cv2
import time
import requests

# URL to fetch bio details
BIO_URL = "http://localhost:8000/get-bio-details"  # Replace with your actual URL if needed


# Function to fetch bio details
def fetch_bio_details():
    try:
        response = requests.get(BIO_URL)
        bio_details = response.json()
        print("Bio details:", bio_details)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bio details: {e}")


# Function for motion detection
def detect_motion():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize the first frame for motion detection
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between current and previous frames
        frame_diff = cv2.absdiff(prev_gray, gray)

        # Threshold the difference to highlight significant changes
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Find contours to detect areas of movement
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are any contours, movement is detected
        if contours:
            print("Movement detected!")
            fetch_bio_details()  # Fetch bio details on movement

        # Update the previous frame
        prev_gray = gray

        # Display the current frame for debugging
        cv2.imshow("Camera Feed", frame)

        # Press 'q' to quit the webcam capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_motion()
