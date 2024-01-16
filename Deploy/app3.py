# server.py

import base64
# import sys
# sys.path.append("D:\\Library\\Documents\\UM Lecture Notes & Tutorial\\FYP\\Src\\Automated-Recognition-of-Facial-Expression-of-Pain")
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
import tensorflow as tf
from mtcnn import MTCNN
from keras.models import load_model
import preprocessData

app = Flask(__name__)

model = load_model("_model\\model_fold_2.h5",custom_objects={"F1Score": tf.keras.metrics.F1Score(
    average="weighted", threshold=None, name='f1_score', dtype=None
)})
print(model.summary())
face_detector = MTCNN()
class_labels = ['None', 'Mild', 'Moderate', 'Very Pain', 'Severe']

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        img_data = request.files['image'].read()
        nparr = np.fromstring(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = face_detector.detect_faces(frame)
        predicted = ""

        if not faces:
             raise ValueError("No face detected in the image")
        if len(faces)> 1:
            raise ValueError("More than 1 subject presented in the camera.")
        for face in faces:
            x, y, w, h = face['box']
            i_frame = [frame]
            face_size = max(w, h)

            if face_size < 150:
                raise ValueError("Subject is too far from the camera.")


            try:
                preprocessing = preprocessData.preprocess(i_frame)
            except Exception as e:
                print(e)

                return jsonify({"result": "error", "message": str(e)})

            input_frame = preprocessing.get_preprocessed_frames()

            # Perform pain prediction using your CNN model
            prediction = model.predict(input_frame)
        
            pred_classes = tf.argmax(prediction, axis=1)
            result = tf.make_ndarray(tf.make_tensor_proto(pred_classes)).tolist()

            color =()
            
            match result[0]:
                case 0:
                    color = (0, 255, 0)
                    
                case 1:
                    color = (255, 255, 0)
                    
                case 2: 
                    color = (0, 255, 255)
                    
                case 3:
                    color = (0, 127, 255)
                    
                case 4:
                    color = (0, 0, 255)

            # Draw bounding box and pain score on the frame
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            frame = cv2.putText(frame, f'Pain Class: {class_labels[result[0]]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            predicted = class_labels[result[0]]
    
        ret, jpeg = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(jpeg).decode('utf-8')

        # Log the result
        log_result(f"Pain Class: {predicted}")

        return jsonify({"result": "success", "image": img_base64, "class": predicted, "value":class_labels.index(predicted)})

    except Exception as e:
        log_result(f"Error: {str(e)}")
        return jsonify({"result": "error", "message": str(e)})

def log_result(message):
    # Log the result to a text area or a file
    print(message)

if __name__ == '__main__':
    app.run(debug=True)
