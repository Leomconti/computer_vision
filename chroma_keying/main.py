import argparse

import cv2
import numpy as np

cap_source = cv2.VideoCapture("data/webcam.mp4")
cap_background = cv2.VideoCapture("data/praia.mp4")

while True:
    # Read a frame from the webcam and the background video
    ret_source, frame_source = cap_source.read()
    ret_background, frame_background = cap_background.read()
    
    # Check if the return is None
    if not ret_source or not ret_background:
        break
    
    # If needed, resize the background to fit the source
    frame_background = cv2.resize(frame_background, (frame_source.shape[1], frame_source.shape[0]))
    
    # Define boundaries for background color
    lower_green = np.array([0, 100, 0], dtype = "uint8")
    upper_green = np.array([100, 255, 100], dtype =  "uint8")
    
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
cap_background.release()
cv2.destroyAllWindows()