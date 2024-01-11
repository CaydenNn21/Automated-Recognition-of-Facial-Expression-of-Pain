# app.py
"""
Initialise the system path
"""
import sys
sys.path.append("D:\\Library\\Documents\\UM Lecture Notes & Tutorial\\FYP\\Src\\Automated-Recognition-of-Facial-Expression-of-Pain")

from flask import Flask, render_template, Response, stream_with_context
from flask_socketio import SocketIO, emit
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from keras.models import load_model
from Module.TestFile import preprocessData
import logging

logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app)

predicted_class = ""

model = load_model("_model\\model_fold_1.h5",custom_objects={"F1Score": tf.keras.metrics.F1Score(
    average="weighted", threshold=None, name='f1_score', dtype=None
)})
face_detector = MTCNN()
class_labels = ['No Pain', 'Mild Pain', 'Moderate Pain', 'Very Pain', 'Severe Pain']


@app.route('/')
def index():
    return render_template('index2.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index
    while True:
        result = []
        global predicted_class
        success, frame = cap.read()
        try:    
            if not success:
                break

              # Face detection using MTCNN
            faces = face_detector.detect_faces(frame)

            for face in faces:
                x, y, w, h = face['box']
                i_frame = [frame]
                preprocessing = preprocessData.preprocess(i_frame)

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'Pain Class: {class_labels[result[0]]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                predicted_class = class_labels[result[0]]
                update_result(predicted_class)
              
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        except Exception as e:
            print(e)
            logging.error(f'Exception in generate_frames: {e}')

            error_message = f'Error: {str(e)}\n'
            write_error(error_message)

    cap.release()

def write_error(error_message):
    byte_data = (b'--frame\r\n'
                 b'Content-Type: text/plain\r\n\r\n' + error_message.encode() + b'\r\n\r\n')
    return Response(byte_data, mimetype='text/event-stream')

@socketio.on('my_event',namespace='/test')
def update_result(output): 
    global current_message
    prediction_message = f'Prediction: {output}\n'
    current_message = prediction_message
    print(current_message)
    # Emit an event to the connected clients with the updated message
    socketio.emit('result', {'message':current_message},namespace='/test')
    
# @socketio.on('update_result',namespace='/test')
# def handle_my_custom_event(json):
#     print('received json: ' + str(json))
#     update_result(predicted_class) # uncomment this line to start sending data to frontend

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True)
