## Code Here
import os
import cv2 as cv
import tensorflow as tf
import torch
from face_alignment import FaceAlignment
from face_alignment import LandmarksType
from preprocessData import preprocess
from keras.models import load_model

import tensorflow as tf
from sklearn.metrics import f1_score

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=1)  # Convert one-hot encoded to class indices
        y_pred = tf.math.argmax(y_pred, axis=1)

        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), tf.float32))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

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
    frame_preprocessing = preprocess(frames)

    input = frame_preprocessing.get_preprocessed_frames()

    print(len(input))

    # Replace 'your_model_path' with the actual path to your saved model file
    loaded_model = tf.keras.models.load_model("C:\\Users\\xiao cheng\\Downloads\\model_fold_1.h5",
                                          custom_objects={"F1Score": F1Score})

    pred = loaded_model.predict(input)
    predicted_classes = tf.argmax(pred, axis=1)

    
