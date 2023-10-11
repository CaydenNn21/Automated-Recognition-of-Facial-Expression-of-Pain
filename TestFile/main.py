## Code Here
import os
import cv2 as cv
import torch
from face_alignment import FaceAlignment
from face_alignment import LandmarksType
from preprocessData import preprocess

def capture_n_display():
    cap = cv.VideoCapture(0)

    # Face detector option can be blazeface, sfd, or dlib (must install with visual studio C++)
    model = FaceAlignment(landmarks_type= LandmarksType.TWO_D, face_detector='blazeface', face_detector_kwargs={'back_model': True},device='cpu')
    total_frames = int (5*30)
    frames = [None]*total_frames
    count = 0 
    while count < total_frames:
        ret, frame = cap.read()
        frames[count] = frame
        count += 1
        cv.imshow('OpenCv',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return frames

if __name__ == '__main__':
    # Prompt Start Option
    # Get the video frames of pain 

    while(True):	

        start = True if (str(input("Start? (Y/n) :"))).lower() == "y" else False

        if (start):
            frames = capture_n_display()
            break
    
     # Preprocess each frame
    preprocess(frames)
    
