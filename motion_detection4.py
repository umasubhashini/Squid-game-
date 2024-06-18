import cv2
import os
from datetime import datetime
import pygame

# Initialize pygame for sound
pygame.init()

# Load sound files
background_sound = pygame.mixer.Sound("voicy_squid Game.mp3")  # Background sound
gunshot_sound = pygame.mixer.Sound("laser-gun-81720.mp3")  # Gunshot sound

# Play the background sound on a continuous loop
background_sound.play(-1)

# Create background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Create HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture video from the camera with CAP_DSHOW backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create a directory to store the captured images
output_directory = "captured_images"
os.makedirs(output_directory, exist_ok=True)

# Set the threshold for small motion
motion_threshold = 500

# Flag to track if a person's motion is detected
person_motion_detected = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply background subtraction for person detection
    fg_mask_people = bg_subtractor.apply(frame)

    # Detect people using HOG
    people_boxes, _ = hog.detectMultiScale(frame)

    # Check if people_boxes is not empty
    if people_boxes is not None and not person_motion_detected:
        for (x, y, w, h) in people_boxes:
            # Create a region of interest for person
            roi = fg_mask_people[y:y+h, x:x+w]

            # Calculate adaptive threshold for the ROI
            _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Check for significant motion around the detected person
            if cv2.countNonZero(roi_threshold) > motion_threshold:
                # Play gunshot sound
                gunshot_sound.play()

                # Draw green box around the detected person
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Capture a snapshot when significant motion is detected around a person
                snapshot = frame.copy()
                cv2.putText(snapshot, "Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Generate a unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = os.path.join(output_directory, f"motion_detected_{timestamp}.jpg")

                # Save the snapshot
                cv2.imwrite(filename, snapshot)
                print(f"Image saved: {filename}")

                # Set the flag to True
                person_motion_detected = True

    # Display the result
    cv2.imshow("Motion Detection Game", frame)

    # Check if 'q' key is pressed to quit the game
    key = cv2.waitKey(1)
    if key == ord('q') or person_motion_detected:
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Stop the background sound and quit Pygame
background_sound.stop()
pygame.quit()
