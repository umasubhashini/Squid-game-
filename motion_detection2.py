import cv2
import os
from datetime import datetime

# Create background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Create HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture video from the camera with CAP_DSHOW backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set up game parameters
player_position = [50, 50]
player_radius = 20
score = 0

# Create a directory to store the captured images
output_directory = "captured_images"
os.makedirs(output_directory, exist_ok=True)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Detect people using HOG
    people_boxes, _ = hog.detectMultiScale(frame)

    # Check if people_boxes is not empty
    if people_boxes is not None:
        # Draw player
        cv2.circle(frame, (player_position[0], player_position[1]), player_radius, (0, 255, 0), -1)

        # Draw rectangles around detected people
        for (x, y, w, h) in people_boxes:
            # Draw a green rectangle outline around detected people
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture a snapshot when a person is detected
            snapshot = frame.copy()
            cv2.putText(snapshot, "Person Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(output_directory, f"person_detected_{timestamp}.jpg")

            # Save the snapshot
            cv2.imwrite(filename, snapshot)
            print(f"Image saved: {filename}")

        # Display the result
        cv2.imshow("Motion Detection Game", frame)

        # Move the player based on keyboard input
        key = cv2.waitKey(1)
        if key == ord('w'):
            player_position[1] -= 5
        elif key == ord('s'):
            player_position[1] += 5
        elif key == ord('a'):
            player_position[0] -= 5
        elif key == ord('d'):
            player_position[0] += 5

        # Check if player is overlapping with detected people
        for (x, y, w, h) in people_boxes:
            if (
                x < player_position[0] < x + w
                and y < player_position[1] < y + h
            ):
                print("Game Over! Your Score:", score)
                break

        # Update score
        score += 1

    # Check if 'q' key is pressed to quit the game
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()