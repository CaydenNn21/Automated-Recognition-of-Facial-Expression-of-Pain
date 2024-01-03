import math
from face_alignment import FaceAlignment
from face_alignment import LandmarksType
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
import os
import torch
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms


class preprocess():
    def __init__(self, frames):
        super(preprocess, self).__init__()
        self.frames = frames
        
        # Select 21 frames from the video sequence for prediction
        # self.frames = self.selectFrame()

        # Save the selected frames to a folder
        self.saveFramestoFiles()

        # Do landmark detection on the input frames for face recognition proposes
        self.landmarkDetection()

        # Mask the non-face area with black pixels
        self.frames = self.maskFace()

        # Tilt and align the face at centre, then crop the frames according to the face region
        self.frames = self.tiltAlign()

        for i in range (len(self.frames)):
            cv.imwrite(f'{"rawFrames"}\cropped_frame_{i}.png', cv.cvtColor(self.frames[i], cv.COLOR_BGR2RGB))

        # self.tensor = self.padding_normalization(24)
        self.normalized_img()
        


    def landmarkDetection(self):
        frames = self.frames
        output = []
        framesLandmark = []
        model = FaceAlignment(landmarks_type=LandmarksType.TWO_D, face_detector='blazeface',
                              face_detector_kwargs={'back_model': True}, device='cpu')
        for n in range(0, len(frames)):
            img = (frames[n])
            img = img.copy()
            landmarks = model.get_landmarks(img)
            landmarks_tuple = []
            if landmarks is not None:
                # Iterate over the detected faces
                for pred in landmarks:
                    # Draw landmarks on the frame
                    for point in pred:
                        x, y = point
                        landmarks_tuple.append((int(x), int(y)))
                        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                            cv.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

            framesLandmark.append(landmarks_tuple)
            output.append(img)
        self.framesLandmark = framesLandmark

    def tiltAlign(self):
        frames = self.frames
        output =[]
        for i in range(len(frames)):
            img = frames[i]
            landmarkTuple = self.framesLandmark[i]
            # Landmark index of reight eye and left eye are
            right_eye_cood = [(landmarkTuple[39][0] + landmarkTuple[36][0])/2, (landmarkTuple[39][1] + landmarkTuple[36][1])/2]
            left_eye_cood = [(landmarkTuple[45][0] + landmarkTuple[42][0])/2, (landmarkTuple[45][1] + landmarkTuple[42][1])/2]
            x1, y1 = right_eye_cood
            x2, y2 = left_eye_cood

            a = abs(y1 - y2)
            b = abs(x2 - x1)
            c = math.sqrt(a * a + b * b)

            cos_alpha = (b * b + c * c - a * a) / (2 * b * c)

            alpha = np.arccos(cos_alpha)
            alpha = (alpha * 180) / math.pi
            img = Image.fromarray(img)
            if y1>y2 :
                alpha = -alpha
            img = np.array(img.rotate(alpha))
            output.append(img)
        return output
    
    def maskFace(self):
        routes = [i for i in range (16,-1,-1)] + [i for i in range (17,26+1)]
        
        frames = self.frames
        output = []
        for n in range(len(frames)):
            routes_cod = []
            mask = None
            out = None
            landmarks_tuple = self.framesLandmark[n]
            img = (frames[n])
            img = img.copy()
            img2 = img.copy()
            for i in range (0, len(routes)-1):
                source_point = routes[i]
                target_point = routes[i+1]
                
                source_cod = landmarks_tuple[source_point]
                target_cod = landmarks_tuple[target_point]
                routes_cod.append(source_cod)
                cv.line(img, (source_cod), (target_cod),(255,255,255),2)

            routes_cod = routes_cod+[routes_cod[0]]

            mask = np.zeros((img.shape[0], img.shape[1]))
            mask = cv.fillConvexPoly(mask, np.array(routes_cod),1)
            mask = mask.astype(np.bool_)
            out = np.zeros_like(img)
            out[mask] = img2[mask]
            # plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
            output.append(cv.cvtColor(self.cropFaceArea(out, mask), cv.COLOR_BGR2RGB))
        return output

    def cropFaceArea(self, frame, mask):

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour
        
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)

        # Crop the image to the size of the masked face
        cropped_image = frame[y:y+h, x:x+w]

        return cropped_image

    def selectFrame(self):
        frames = self.frames
        return [frames[i] for i in range(0, 150, 7)]

    def saveFramestoFiles(self):
        frames = self.frames

        if not os.path.exists("rawFrames"):
            os.mkdir("rawFrames")

        for i in range(len(frames)):
            cv.imwrite(f'{"rawFrames"}\selectedFrames_{i}.png', frames[i])



    def padding_normalization(self, target_length):
        """
        Preprocesses a sequence of images and pads them to a target length.

        Args:
            images (list): List of PIL images.
            target_length (int): Desired length of the sequence after padding.

        Returns:
            torch.Tensor: Tensor of preprocessed and padded images.
        """
        # Resize the images to a consistent size
        array_images = self.frames
        images =[]

        for image in array_images:
            images.append((Image.fromarray(image)))

        resized_images = [TF.resize((img), [150, 150]) for img in images]

        # Convert the images to tensors
        tensor_images = [TF.to_tensor(img) for img in resized_images]

        # Stack the tensor images along a new dimension (sequence dimension)
        tensor_sequence = torch.stack(tensor_images)

        # Calculate the current length of the sequence
        current_length = tensor_sequence.size(0)

        # Pad the sequence if necessary
        if current_length < target_length:
            padding_length = target_length - current_length
            padding = torch.zeros(padding_length, *tensor_sequence.shape[1:])
            tensor_sequence = torch.cat((tensor_sequence, padding))

        # Normalize the tensor sequence
        # Define the mean and standard deviation values for normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Apply normalization to the tensor sequence
        normalize = transforms.Normalize(mean=mean, std=std)
        self.normalized_sequence = normalize(tensor_sequence)

        return self.normalized_sequence
    
    def normalized_img(self):
        frames_temp = []
        for image in self.frames:
            frames_temp.append(cv.resize(image, dsize=(150, 150), interpolation=cv.INTER_CUBIC))

        all_frames32 = np.array(frames_temp, dtype="float32")
        # Normalize the frames
        all_frames_1 = all_frames32/ 255.0
        self.normalized_frames = all_frames_1
        return all_frames_1

    
    def get_preprocessed_frames(self):
        frames = self.normalized_frames
        return frames
