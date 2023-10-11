import cv2 as cv
import torch
from face_alignment import FaceAlignment
from face_alignment import LandmarksType


cap = cv.VideoCapture(0)

# Face detector option can be blazeface, sfd, or dlib (must install with visual studio C++)
model = FaceAlignment(landmarks_type= LandmarksType.TWO_D, face_detector='blazeface', face_detector_kwargs={'back_model': True},device='cpu')
total_frames = int (5*30)
count = 0 
while count <= total_frames:
    ret, frame = cap.read()
    faces = model.get_landmarks(frame)

    if faces is not None:
        # Iterate over the detected faces
        for pred in faces:
            # Draw landmarks on the frame
            for point in pred:
                x, y = point
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    count += 1
    cv.imshow('OpenCv',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    