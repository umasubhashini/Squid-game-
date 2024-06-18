import cv2
import numpy as np

# Function to calculate absolute difference between two frames
def absdiff(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return threshold

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Read two consecutive frames to initialize the loop
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    # Get the absolute difference between two frames
    threshold = absdiff(frame1, frame2)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Motion Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
