import argparse

import cv2
import numpy as np

cap_source = cv2.VideoCapture("data/com.mp4")
# the background will be static as it's a jpeg image, so we use imread and use the same frame for all the video
frame_background = cv2.imread("data/background_office.jpeg")

while True:
    # Read a frame from the webcam
    ret_source, frame_source = cap_source.read()
    # Check if the return is None
    if not ret_source:
        break
    
    # If needed, resize the background to fit the source
    frame_background = cv2.resize(frame_background, (frame_source.shape[1], frame_source.shape[0]))
    
    # Define the background color as with the video matting with have a stable green color for the background
    lower_green = np.array([0, 150, 40], dtype="uint8")
    upper_green = np.array([180, 255, 150], dtype="uint8")

    # Create mask for background color
    mask = cv2.inRange(frame_source, lower_green, upper_green) 
    
    # Use the mask to put the background video in the pixels related to the background
    background = cv2.bitwise_and(frame_background, frame_background, mask = mask)
     
    # Inverts the mask to get the pixels related to the foreground
    mask_inv = np.invert(mask)
    
    # Use the inverted mask to extract the pixels related to the foreground
    foreground = cv2.bitwise_and(frame_source, frame_source, mask = mask_inv)
    
    # Combines the foreground of the webcam with the background video
    result =  cv2.addWeighted(foreground, 1, background, 1, 0)
    
    # Display the resulting frame
    cv2.imshow('result', result)
    
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When finished release the capture and destroy all windows
cap_source.release()
cv2.destroyAllWindows()