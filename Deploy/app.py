
import base64
import sys

import cv2
sys.path.append("D:\\Library\\Documents\\UM Lecture Notes & Tutorial\\FYP\\SourceCode")
import io
from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from Module.TestFile.F1Score import F1Score
from Module.TestFile import preprocessData
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = load_model("C:\\Users\\xiao cheng\\Downloads\\model_fold_1.h5",custom_objects={"F1Score": F1Score})

# Define the class labels (modify based on your classes)
class_labels = ['No Pain', 'Mild Pain', 'Moderate Pain', 'Very Pain', 'Severe Pain']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("predicting")
    try:
        # Get the image data from the POST request
        image_data = request.get_json()['captured_frames']
        converted = convert_image(image_data)
        imgs = preprocess_input(converted)

        # Make a prediction
        prediction = model.predict(imgs)
        
        pred_classes = tf.argmax(prediction, axis=1)
        result = tf.make_ndarray(tf.make_tensor_proto(pred_classes)).tolist()
        print(result)
        class_idx = int(max_occurrences(result))
        print(class_idx)
        predicted_class = class_labels[class_idx]

        return jsonify({'class': predicted_class, 'confidence': float(np.max(prediction))})

    except Exception as e:
        return jsonify({'error': str(e)})
    
def max_occurrences(List):

    return max(set(List), key = List.count)

def preprocess_input(frames):
    preprocess_imgs = preprocessData.preprocess(frames)
    imgs = preprocess_imgs.get_preprocessed_frames()
    return imgs

def convert_image(data):
    temp = []
    for bytecode in data:
        # Convert base64 image data to bytes
        img_bytes = base64.b64decode(bytecode.split(',')[1])

        # Convert bytes to numpy array
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)

        # Decode the image
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        temp.append(np.array(img))

    
    return temp

if __name__ == '__main__':
    app.run(debug=True)
